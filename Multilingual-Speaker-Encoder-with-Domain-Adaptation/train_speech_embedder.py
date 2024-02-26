#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import time
import librosa
import subprocess
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from os.path import join, exists, basename

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDataset, Collate
from speech_embedder_net import SpeechEmbedder, GE2ELoss, Classifier
from utils import get_centroids, get_cossim, accuracy, count_label, get_classifier_loss, compute_da_threshold
from module import MultiLayerNN

def process_dvector_wav(wav, mel_calculator):
    #utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    min_seg_length = 130
    utter_min_len = int((min_seg_length * hp.data.hop + hp.data.window) * hp.data.sr)    # lower bound of utterance length
    # if the wav is shorter than minimal length, we don't split the wav
    if utter_min_len >= wav.shape[0]:
        intervals = [(0, wav.shape[0])]
    else:
        intervals = librosa.effects.split(wav, top_db=100)
    if len(intervals) == 0:
        return None
    specs = []
    for interval in intervals:
        inter_length  = interval[1]-interval[0]
        if inter_length < utter_min_len:
            if inter_length <= utter_min_len // 4 and len(intervals) >= 2:
                continue
            print(wav.shape[0], inter_length)
            utter_part = np.pad(wav[interval[0]:interval[1]], ((0, utter_min_len - inter_length)), mode='constant', constant_values=0)
        else:
            utter_part = wav[interval[0]:interval[1]]
        S = mel_calculator.get_mel(utter_part) 
        seg_length = min_seg_length
        step = seg_length // 2
        for i in range(0, S.shape[1], step):
            if i + seg_length > S.shape[1]:
                break
            specs.append(S[:, i:i+seg_length])
    if not specs:
        return None
    else:
        return np.array(specs)

def load_embedder(path, device):
    model = SpeechEmbedder().to(device)
    load_state_dict(model, None, path)
    return model

def generate_dvector(wav, model, device, spec_path, mel_calculator):
    if not exists(spec_path):
        specs = process_dvector_wav(wav, mel_calculator)
        specs = np.transpose(specs, axes=(0,2,1))
        torchspecs = torch.tensor(specs).to(device)
    else:
        torchspecs = torch.load(spec_path).to(device)
        specs = torchspecs
    if specs is None:
        return None, None
    out_dict = model(torchspecs)
    embedding = out_dict.get('embeddings')
    embedding = embedding.mean(0).squeeze()
    embedding = embedding / embedding.norm()
    embedding = embedding.detach().cpu()
    torchspecs = torchspecs.cpu()
    return embedding, torchspecs

def lr_schedule(optimizers, epoch, optimizable):
    if optimizable is None:
        optimizable = []
    if epoch in hp.train.anneal_epochs:
        for k, optimizer in optimizers.items():
            if k not in optimizable:
                continue
            old_lr = optimizer.param_groups[0]['lr']
            lr = old_lr / 2
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            print('anneal learning rate, {} -> {}'.format(old_lr, lr))
        print(optimizers)

def load_state_dict(net, loss, path):
    print('load state dict from {}'.format(path))
    sd = torch.load(path)
    net.load_state_dict(sd['embedder_net'])
    if loss is not None:
        loss.load_state_dict(sd['ge2e'])
    
def initialize_optimizers(embedder_net, ge2e_loss):
    optimizers = {}
    optim = getattr(torch.optim, hp.train.optimizer)
    optimizer = optim(embedder_net.main_parameters(), lr=hp.train.lr, weight_decay=1e-6)
    optimizers['main']  = optimizer
    optimizers['ge2e'] = optim(ge2e_loss.parameters(), lr=hp.train.lr)
    if hp.model.da:
        da_optimizer = optim(embedder_net.da_parameters(), 1e-3, weight_decay=1e-6)
        optimizers['da'] = da_optimizer
    return optimizers

def train(model_path):
    device = torch.device(hp.device)
    
    if hp.data.data_preprocessed:
        train_dataset = SpeakerDataset()
    else:
        train_dataset = SpeakerDatasetTIMIT()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=False,
                    num_workers=hp.train.num_workers, drop_last=True,
                    collate_fn=Collate(True)) 
    
    embedder_net = SpeechEmbedder().to(device)
    print(embedder_net)
    if hp.train.parallel:
        dist.init_process_group()
        embedder_net_run = DDP(embedder_net)
    else:
        embedder_net_run = embedder_net
    ge2e_loss = GE2ELoss(device)
    if hp.train.restore:
        load_state_dict(embedder_net, ge2e_loss, model_path)
    #Both net and loss have trainable parameters
    optimizers = initialize_optimizers(embedder_net, ge2e_loss)
        
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    embedder_net.train()
    iteration = 0
    loss_buf = []
    start_epoch = hp.train.start_epoch if hp.train.restore else 0
    da_opt_threshold = compute_da_threshold(hp)
    for e in range(start_epoch, hp.train.epochs):
        total_loss = 0
        total_da_loss = 0
        total_da_acc = 0
        btime = time.time()
        train_progress = float(e) / hp.train.epochs
        if hp.model.da:
            embedder_net.adjust_da_coef(train_progress)
        for batch_id, batch in enumerate(train_loader): 
            mel_db_batch = batch.get('mels')
            langs = batch.get('langs')
            mel_db_batch = mel_db_batch.to(device)
            langs = langs.to(device)
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = list(range(0, hp.train.N*hp.train.M))
            random.shuffle(perm)
            unperm = list(range(0, hp.train.N*hp.train.M))
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #get loss
            zero_grad(optimizers, ['main', 'ge2e', 'da'])
            out_dict = embedder_net_run(mel_db_batch)
            embeddings = out_dict.get('embeddings')
            da_lang_logits = out_dict.get('da_lang_logits')
            embeddings = embeddings[unperm]
            da_lang_logits = da_lang_logits[unperm] if da_lang_logits is not None else da_lang_logits
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            loss_params = {}
            
            # accumulate gradient
            loss_sum, loss, da_loss = ge2e_loss(embeddings, da_lang_logits, langs, **loss_params) #wants (Speaker, Utterances, embedding)
            acc = accuracy(da_lang_logits, langs, binary=count_label(hp) == 1)
            da_threshold = da_loss < da_opt_threshold
            da_pretrain = train_progress  <= hp.model.da_startpoint
            # loss.backward(retain_graph = (hp.model.da and (da_threshold or da_pretrain)))

            if hp.model.da and (da_threshold or da_pretrain):
                da_loss.backward()
            # step
            torch.nn.utils.clip_grad_norm_(embedder_net.main_parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optim_step(optimizers, ['main', 'ge2e'])
            if hp.model.da and (da_threshold or da_pretrain):
                torch.nn.utils.clip_grad_norm_(embedder_net.da_parameters(), 3.0)
                optimizers['da'].step()
            
            total_loss = total_loss + loss
            total_da_loss = total_da_loss + da_loss
            total_da_acc = total_da_acc + acc
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.2f}".format(time.ctime(), e+1, batch_id+1, len(train_dataset)//hp.train.N, iteration, loss)
                if hp.model.da:
                    mesg += "\tDALoss:{0:.2f}\tDAAcc:{1:.2f}".format(da_loss, acc)
                mesg += '\n'
                print(mesg)
                #if hp.train.log_file is not None:
                    #with open(hp.train.log_file,'a') as f:
                        #f.write(mesg)
        etime = time.time()
        lr_schedule(optimizers, e, ['main', 'ge2e'])
        avg_loss = float(total_loss) / (batch_id + 1)
        avg_da_loss = float(total_da_loss) / (batch_id + 1)
        avg_da_acc = float(total_da_acc) / (batch_id + 1)
        print('Epoch {} end, avg loss: {}, avg da loss: {}, avg da accuracy: {}, time: {}s'.format(e + 1, avg_loss, avg_da_loss, avg_da_acc, etime - btime))
        state = {'loss': avg_loss, 'da_loss': avg_da_loss, 'da_acc': avg_da_acc}
        if hp.model.da and not da_pretrain:
            da_classifier_subroutine(embedder_net, train_loader, optimizers['da'])

        loss_buf.append(state)
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_{}_epoch_".format(hp.model.model_name) + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
            ckpt_model_path = join(hp.train.checkpoint_dir, ckpt_model_filename)
            sd = {'embedder_net': embedder_net.state_dict(), 'ge2e': ge2e_loss.state_dict()}
            torch.save(sd, ckpt_model_path)
            embedder_net.to(device).train()

    save_model_filename = "{}_{}".format(hp.model.model_name, e+1)
    # before save model, first we generate embedding for each speaker
    spkr2embedding = get_spkr2embedding(embedder_net, train_dataset)
    torch.save(spkr2embedding, join(hp.train.checkpoint_dir, save_model_filename + ".embedding"))
    visualize_embeddings(spkr2embedding)

    #save model
    embedder_net.eval().cpu()
    save_model_path = join(hp.train.checkpoint_dir, save_model_filename + ".model")
    sd = {'embedder_net': embedder_net.state_dict(), 'ge2e': ge2e_loss.state_dict()}
    torch.save(sd, save_model_path)
    print("\nDone, trained model saved at", save_model_path)

    # plot loss
    keys = list(loss_buf[0].keys())
    for k in keys:
        data = [item[k] for item in loss_buf]
        plt.plot(data)
        f = plt.gcf()
        f.savefig('./result/{}_{}.png'.format(hp.model.model_name, k))
        plt.clf()
    torch.save(loss_buf, join(hp.train.checkpoint_dir, save_model_filename + ".loss"))


def da_classifier_subroutine(model, train_loader, optimizer):
    device = torch.device(hp.device)
    loss_funct = get_classifier_loss(hp)
    prev_loss = float('inf')
    for e in range(10):
        avg_loss = avg_acc = 0
        for batch_id, batch in enumerate(train_loader):
            mel_db_batch = batch.get('mels')
            mel_db_batch = mel_db_batch.to(device)
            langs = batch.get('langs')
            langs = langs.to(device)
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = list(range(0, hp.train.N*hp.train.M))
            random.shuffle(perm)
            unperm = list(range(0, hp.train.N*hp.train.M))
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            optimizer.zero_grad()
            # we set detach to speed up the gradient computation
            outdict = model(mel_db_batch, detach=True)
            embeddings = outdict.get('embeddings')[unperm]
            lang_logits = outdict.get('da_lang_logits')
            lang_logits = lang_logits[unperm] if lang_logits is not None else lang_logits
            acc = accuracy(lang_logits, langs, binary=count_label(hp) == 1)
            da_loss = loss_funct(lang_logits, langs, reduction='sum')
            da_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.da_classifier.parameters(), 3.0)
            optimizer.step()
            avg_loss += float(da_loss)
            avg_acc += acc
        avg_loss /= batch_id + 1
        avg_acc /= batch_id + 1
        print('Da subroutine epoch {}, avg loss: {}, avg acc: {}'.format(e, avg_loss, avg_acc))
        if avg_loss < 20:
            break
        if avg_loss > prev_loss:
            break
        prev_loss = avg_loss

def get_spkr2embedding(embedder_net, dataset):
    dataset.shuffle = False
    dataset.utter_num = 400
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                    num_workers=hp.train.num_workers, drop_last=False,
                    collate_fn=Collate()) 
    device = torch.device(hp.device)
    spkr2embedding = {}
    embedder_net.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            mel = batch.get('mels')
            mel = mel.to(device)
            mel = mel.squeeze()
            out_dict = embedder_net(mel)
            embeddings = out_dict.get('embeddings')
            spkr = dataset.file_list[i]
            print(i, spkr, mel.shape)
            spkr2embedding[spkr] = embeddings
    embedder_net.train()
    return spkr2embedding

def infer_embedding(path):
    emb_path = join(hp.train.checkpoint_dir, hp.model.model_name+'_{}.embedding'.format(hp.train.epochs))
    if not exists(emb_path):
        device = torch.device(hp.device)
        if hp.data.data_preprocessed:
            train_dataset = SpeakerDataset()
        else:
            train_dataset = SpeakerDatasetTIMIT()
    
        embedder_net = SpeechEmbedder().to(device)
        #if hp.train.restore:
        load_state_dict(embedder_net, None, path)
        spkr2embedding = get_spkr2embedding(embedder_net, train_dataset)
        torch.save(spkr2embedding, emb_path)
    else:
        spkr2embedding = torch.load(emb_path)
    visualize_embeddings(spkr2embedding)

def visualize_embeddings(spkr2embedding):
    import matplotlib.patches as mpatches
    spkrs, embs = zip(*list(spkr2embedding.items()))
    emb_per_spkr = embs[0].shape[0]
    embs = torch.stack(embs)
    assert embs.shape[0] * embs.shape[1] == emb_per_spkr * len(spkrs), "total embedding number {} is not equal to emb_per_spkr {} times length of spkr {}".format(embs.shape[0], emb_per_spkr, len(spkrs))
    # sample the spkr
    infos = [(i, SpeakerDataset.decode_filename(s)) for i, s in enumerate(spkrs)]
    lang2indices = {}
    for index, info in infos:
        l = info[hp.model.da_on]
        if l not in lang2indices:
            lang2indices[l] = []
        lang2indices[info[hp.model.da_on]].append(index)
    spkr_per_lang = 300
    new_emb_per_spkr = 400
    for l, v in lang2indices.items():
        lang2indices[l] = random.sample(v, min(spkr_per_lang, len(v)))
    lang2indices = list(lang2indices.items())
    langs = []
    for l, idx in lang2indices:
        langs.extend([l] * len(idx))
    lang2code = list(set(langs))
    lang2code.sort()
    lang2code = {k:i for i, k in enumerate(lang2code)}
    print(lang2code)
    indices = [i for item in lang2indices for i in item[1]]
    embs = embs[indices, :new_emb_per_spkr, :]
    #embs = embs.reshape(len(indices) * new_emb_per_spkr, -1)
    embs = embs.mean(1)
    print(embs.shape)
    embs = embs.cpu().numpy()
    
    print('tsne fit...')
    tsne = TSNE(2, perplexity=50, init='pca', method='exact')
    out = tsne.fit_transform(embs)
    print('tsne done')
    out = (out - out.min(0)) / (out.max(0) - out.min(0))
    #out = out.reshape(len(indices), new_emb_per_spkr, -1)
    plt.figure(figsize=(9.84, 7.47))
    plt.tick_params(labelsize=28)
    #colors = cm.rainbow(np.linspace(0, 1, len(indices)))
    colors = 'rbyg'
    markers = 'ox^*'
    for i in range(len(indices)):
        marker = markers[lang2code[langs[i]]]
        spkr_emb = out[i]
        plt.scatter(spkr_emb[0], spkr_emb[1], marker=marker, s=32, color=colors[lang2code[langs[i]]], alpha=0.6)
    patches = []
    for k, v in lang2code.items():
        m, c = markers[v], colors[v]
        patch = plt.plot([], [], marker=m, ls='', mec=None, color=c, label=k, alpha=0.6, ms=18)[0]
        patches.append(patch)
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 18})
    f = plt.gcf()
    f.savefig('./result/{}_embedding.png'.format(hp.model.model_name))
    plt.clf()

def test(model_path):
    
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDataset()
    else:
        test_dataset = SpeakerDatasetTIMIT()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N,
                    shuffle=False, num_workers=hp.test.num_workers,
                    drop_last=True, collate_fn=Collate(True))
    
    embedder_net = SpeechEmbedder()
    load_state_dict(embedder_net, None, model_path)
    embedder_net.eval()
    
    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, batch in enumerate(test_loader):
            mel_db_batch = batch.get('mels')
            lang = batch.get('langs')
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            enroll_dict = embedder_net(enrollment_batch)
            enrollment_embeddings, lang_logits = enroll_dict.get('embeddings'), enroll_dict.get('lang_logits')
            verifi_dict = embedder_net(verification_batch)
            verification_embeddings = verifi_dict.get('embeddings')
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            
            # calculating EER
            diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
            
            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix>thres
                
                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)
    
                FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(float(hp.test.M/2))/hp.test.N)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))


def zero_grad(optimizers, keys):
    for k in keys:
        if k in optimizers:
            optimizers[k].zero_grad()

def optim_step(optimizers, keys):
    for k in keys:
        optimizers[k].step()
        
if __name__=="__main__":
    epoch = hp.train.start_epoch if hp.train.restore else hp.train.epochs
    path = join(hp.train.checkpoint_dir, hp.model.model_name+'_{}.model'.format(epoch))
    if hp.training:
        cmd = 'cp ./config/config.yaml {}'.format(join(hp.train.checkpoint_dir, hp.model.model_name+'_{}.config'.format(epoch)))
        subprocess.run(cmd, shell=True)
        train(path)
    else:
        test(path)
