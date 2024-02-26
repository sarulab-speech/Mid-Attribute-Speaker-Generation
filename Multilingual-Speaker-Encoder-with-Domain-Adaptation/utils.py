#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018

@author: harry
"""
import librosa
import numpy as np
import torch
import torch.autograd as grad
import torch.nn.functional as F

from .hparam import hparam as hp

def calculate_centroid_include_self(embedding):
    '''
    calculate centroid embedding. For each embedding, include itself inside the calculation.
    :param embedding: shape -> (N, M, feature_dim)
    :return:
    embedding_mean: shape -> (M, feature_dim)
    '''
    N, M, feature_dim = embedding.shape
    embedding_mean = torch.mean(embedding, dim=1)
    return embedding_mean

def calculate_centroid_exclude_self(embedding):
    '''
    calculate centroid embedding. For each embedding, exclude itself inside the calculation.
    :param embedding: shape -> (N, M, feature_dim)
    :return:
    embedding_mean: shape -> (N, M, feature_dim)
    '''
    N, M, feature_dim = embedding.shape
    embedding_sum = torch.sum(embedding, dim=1, keepdim=True) # shape -> (N, 1, feature_dim)
    embedding_mean = (embedding_sum - embedding) / (M-1)
    return embedding_mean

def calculate_similarity(embedding, centroid_embedding):
    '''
    calculate similarity S_jik
    :param embedding: shape -> (N, M, feature_dim)
    :param centroid_embedding: -> (N, feature_dim)
    :return:
    similarity: shape -> (N, M, N)
    '''
    N, M, feature_dim = embedding.shape
    N_c, feature_dim_c = centroid_embedding.shape
    assert N == N_c and feature_dim == feature_dim_c, "dimension wrong in get_similarity_include_self!"

    centroid_embedding = centroid_embedding.unsqueeze(0).unsqueeze(0).expand(N, M, -1, -1)
    assert centroid_embedding.shape == (N, M, N, feature_dim), "centroid embedding has wrong expansion in get_similarity_include_self."
    embedding = embedding.unsqueeze(2)
    similarity = F.cosine_similarity(embedding, centroid_embedding, dim=3)
    return similarity

def calculate_similarity_j_equal_k(embedding, centroid_embedding):
    '''
    calculate cimilarity S_jik for j == k
    :param embedding: shape -> (N, M, feature)
    :param centroid_embedding: shape -> (N, M, feature)
    :return:
    similarity: shape -> (N, M)
    '''
    N, M, feature_dim = embedding.shape
    N_c, M_c, feature_dim_c = centroid_embedding.shape
    assert N==N_c and M==M_c and feature_dim==feature_dim_c, "dimension wrong in get_similarity_exclude_self!"

    similarity = F.cosine_similarity(embedding, centroid_embedding, dim=2)
    return similarity

def combine_similarity(similarity, similarity_j_equal_k):
    same_index = list(range(similarity.shape[0]))
    similarity[same_index, :, same_index] = similarity_j_equal_k[same_index, :]
    return similarity

def get_similarity(embedding):
    '''
    get similarity for input embedding
    :param embedding: shape -> (N, M, feature)
    :return:
    similarity: shape -> (N, M, N)
    '''
    embedding_mean_include = calculate_centroid_include_self(embedding)
    embedding_mean_exclude = calculate_centroid_exclude_self(embedding)

    similarity = calculate_similarity(embedding, embedding_mean_include) # shape (N, M, N)
    similarity_j_equal_k = calculate_similarity_j_equal_k(embedding, embedding_mean_exclude) # shape (N, M)
    similarity = combine_similarity(similarity, similarity_j_equal_k)
    return similarity

def get_similarity_eva(enrollment_embedding, evaluation_embedding):
    '''
    get similarity score for evaluation
    :param enrollment_embedding: shape -> (N, M_1, feature_dim)
    :param evaluation_embedding: shape -> (N, M_2, feature_dim)
    :return:
    similarity: shape -> (N, M_2, N)
    '''

    enrollment_embedding_mean = calculate_centroid_include_self(enrollment_embedding) # shape -> (N, feature_dim)
    similarity = calculate_similarity(evaluation_embedding, enrollment_embedding_mean) # shape (N, M_2, N)
    return similarity


def get_contrast_loss(similarity):
    '''
    L(e_ji) = 1-sigmoid(S_jij)+max_k(sigmoid(S_jik))
    :param similarity: shape -> (N, M, N)
    :return:
    loss = sum_ji(L(e_ji))
    '''

    # some inplace operation
    # one of the variables needed for gradient computation has been modified by an inplace operation
    # so I choose to implement myself
    sigmoid = 1 / (1 + torch.exp(-similarity))
    same_index = list(range(similarity.shape[0]))
    loss_1 = torch.sum(1-sigmoid[same_index, :, same_index])
    sigmoid[same_index, :, same_index] = 0
    loss_2 = torch.sum(torch.max(sigmoid, dim=2)[0])

    loss = loss_1 + loss_2
    return loss

def get_softmax_loss(similarity):
    '''
    L(e_ji) = -S_jij) + log(sum_k(exp(S_jik))
    :param similarity: shape -> (N, M, N)
    :return:
    loss = sum_ji(L(e_ji))
    '''
    same_index = list(range(similarity.shape[0]))
    loss = torch.sum(torch.log(torch.sum(torch.exp(similarity), dim=2) + 1e-6)) - torch.sum(similarity[same_index, :, same_index])
    return loss

def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):    
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
    
    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)
        
    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)
    
    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    mag_db = librosa.amplitude_to_db(mag_spec)
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T
    
    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T
    
    return mfccs, mel_db, mag_db

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids

def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1)) # (M*N, E)
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1) # (M*N, 1, E) -> (M*N, M, E)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff

def accuracy(x, y, binary=False, percent=True):
    if x is None or y is None or type(x) is int:
        return 0
    if not binary:
        return (torch.argmax(x, 1) == y).sum() / float(y.shape[0]) * (100 if percent else 1)
    else:
        # x should indicate label 1 prob
        label = torch.sigmoid(x).round().long().squeeze()
        out = (label == y).sum() / float(y.shape[0]) * (100 if percent else 1)
        return float(out)

def count_label(hp):
    if hp.model.da_on == 'language':
        return 1

def get_classifier_loss(hp):
    label = count_label(hp)
    if label == 1:
        return F.binary_cross_entropy_with_logits
    else:
        return F.cross_entropy

def compute_da_threshold(hp):
    from math import log, e
    label = count_label(hp)
    if label == 1: label += 1
    return -log(1/label) * hp.train.N * hp.train.M

def mel_spectrogram(wav, hp):
    S = librosa.core.stft(y=wav, n_fft=hp.data.nfft,
                                win_length=int(hp.data.window * hp.data.sr), hop_length=int(hp.data.hop * hp.data.sr))
    S = np.abs(S)
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels, fmin=55, fmax=8000)
    S = np.dot(mel_basis, S)
    S = np.clip(S, 1e-5, None)
    S = np.log(S)
    return S

def mel_spectrogram_old(wav):
    S = librosa.core.stft(y=wav, n_fft=hp.data.nfft,
                                win_length=int(hp.data.window * hp.data.sr), hop_length=int(hp.data.hop * hp.data.sr))
    S = np.abs(S) ** 2
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel 
    return S

if __name__ == "__main__":
    pass
