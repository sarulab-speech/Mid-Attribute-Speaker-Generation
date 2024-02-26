import argparse
import os
from pathlib import Path

import math
import copy
import random
import torch
from torch.nn.parallel.data_parallel import data_parallel
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample, plot_alignment_to_numpy
from model import FastSpeech2Loss, SpeakerMetaEncLoss, GANLike, FastSpeech2, ScheduledOptim
_temp = __import__("Multilingual-Speaker-Encoder-with-Domain-Adaptation", globals(), locals(), ['SpeechEmbedder', 'GE2ELoss'], 0)
SpeechEmbedder = _temp.SpeechEmbedder
GE2ELoss = _temp.GE2ELoss
from dataset import Dataset, ConcatDataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs, corpuses, checkpoint=None):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs


    list_dataset = []
    for corpus in corpuses:
        preprocess_config_ = corpus
        preprocess_config_["preprocessing"] = preprocess_config
        preprocess_config_["preprocessing"]["text"] = corpus["text"]
        preprocess_config_["preprocessing"]["accent"] = corpus["accent"]
        # use accent info?
        use_accent = preprocess_config_['preprocessing']["accent"]["use_accent"]

        # Get dataset
        dataset = Dataset(
            "train.txt", preprocess_config_, train_config, sort=True, drop_last=True
        )
        list_dataset.append(dataset)

    dataset = ConcatDataset(args.config, list_dataset)

    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=20,
        pin_memory=True
    )

    # Prepare model
    model=FastSpeech2(preprocess_config, model_config, args.config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
    discriminator=SpeechEmbedder().to(device)
    if train_config['dataparallel']:
        model = nn.DataParallel(model)
        discriminator = nn.DataParallel(discriminator)
    if checkpoint != None:
        param = torch.load(checkpoint)
        model.load_state_dict(param["model"])
        if "discriminator" in param:
            discriminator.load_state_dict(param["discriminator"])
    
    ganlike = GANLike(model, discriminator, train_config)
    optimizer = ScheduledOptim(ganlike, train_config, model_config, args.restore_step)
    if args.restore_step:
        optimizer.load_state_dict(ckpt["optimizer"])
    ganlike.train()
    num_param = get_param_num(model) + get_param_num(discriminator)

    # Prepare Loss
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    eLoss = SpeakerMetaEncLoss(preprocess_config, model_config).to(device)
    dLoss = GE2ELoss(device)

    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    crosscase = 0
    allcase = 0
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    use_jdit = model_config['jdit']['use_jdit']
    #For now distinguish by multi_speaker option
    use_speaker_gen = model_config['multi_speaker']

    lambd = 1
    if "lambda" in train_config:
        lambd = train_config["lambda"]
    dloss = None
    

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                output, langs, output_r, max_len_r = ganlike(device, batch)

                if use_jdit:
                    losses = Loss(batch[:-2], output[:-4])
                    alignment = batch[-3]
                    total_loss = losses[0]
                    total_loss += nn.MSELoss()(output[-2],batch[6])
                else:
                    losses = Loss(batch[:-2], output[:-2])
                    total_loss = losses[0]
                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                #Cal Loss for speaker_enc
                if use_speaker_gen:
                    eloss = eLoss(output[-1], output[-2])
                    eloss = -eloss / grad_acc_step
                    eloss.backward()

                langs = langs + torch.rand(langs.shape).to(device)
                _, _, dloss = dLoss(output_r.get("embeddings").view(batch_size*max_len_r, 1, -1), output_r.get("da_lang_logits"), langs, reduction='sum')
                # Backward
                # multiply loss with lambda
                dloss_ = dloss * (2 / (1 + math.exp(-10 * (step / total_step))) - 1) / len(langs)
                dloss_ = dloss_ * lambd
                dloss_.backward()


                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, eloss=eloss, dloss=dloss)

                if step % synth_step == 0:
                    if use_jdit:
                        train_logger.add_image(
                            "alignment",
                            plot_alignment_to_numpy(alignment[0].data.cpu().numpy().T),
                            step, 
                            dataformats='HWC'
                        )
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        args,
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(args, model, step, configs, corpuses, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    if train_config['dataparallel']:
                        state_dict = model.module.state_dict()
                        state_dict_disc = discriminator.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                        state_dict_disc = discriminator.state_dict()
                    torch.save(
                        {
                            "model": state_dict,
                            "discriminator": state_dict_disc,
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to config",
    )
    parser.add_argument(
        "--use_clf",
        action="store_true"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="path to checkpoint.pth.tar",
    )
    parser.add_argument(
        "--corpus",
        nargs="*",
        type=str,
        required=False,
        default=None,
        help="set name of corpus if only use some corpuses"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.config + "/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.config + "/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.config + "/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    checkpoint = args.checkpoint
    config_path = Path(args.config)
    if args.corpus != None:
        corpuses = []
        for corpus in args.corpus:
            corpuses.append(config_path / ("preprocess_" + corpus + ".yaml"))
    else:
        corpuses = config_path.glob("preprocess_*.yaml")
    corpuses = [yaml.load(open(co, "r"), Loader=yaml.FullLoader) for co in corpuses]

    main(args, configs, corpuses, checkpoint)
