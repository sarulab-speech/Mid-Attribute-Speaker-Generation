import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss, SpeakerMetaEncLoss
from dataset import Dataset, ConcatDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args, model, step, configs, corpuses, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    list_dataset = []
    for corpus in corpuses:
        preprocess_config_ = corpus
        preprocess_config_["preprocessing"] = preprocess_config
        preprocess_config_["preprocessing"]["text"] = corpus["text"]
        preprocess_config_["preprocessing"]["accent"] = corpus["accent"]
        # use accent info?
        use_accent = preprocess_config_['preprocessing']["accent"]["use_accent"]
        use_speaker_gen = model_config['multi_speaker']

        # Get dataset
        dataset = Dataset(
            "val.txt", preprocess_config_, train_config, sort=False, drop_last=False
        )
        list_dataset.append(dataset)
    
    dataset = ConcatDataset(args.config, list_dataset)


    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    eLoss = SpeakerMetaEncLoss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    eloss_sums = 0
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                accents = batch[-1]
                speaker_meta = batch[-2]
                batch = batch[:-2]
                output = model(*(batch[2:]),accents=accents, speaker_meta=speaker_meta)
                losses = Loss(batch, output[:-2])

                # Cal Loss

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
                if use_speaker_gen:
                    eloss = eLoss(output[-1], output[-2])
                    eloss_sums += eloss * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    eloss_means = eloss_sums / len(dataset)

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )
    if use_speaker_gen:
        message += ", Speaker Loss: {:.4f}".format(eloss_means)

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            args,
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)
