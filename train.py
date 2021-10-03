#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

import experiments.utils as experiment_utils

from tqdm.auto import tqdm
import models.utils as model_utils


def initialise(model, start_epoch=1, folder_name="", lr=0.0001, beta=1.0, l2_reg=1e-4, train_batch_size=100,
                figs_checkpoints_save_freq=10, warmup_time=0, beta_final=1.0):
    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    model.folder_name = folder_name
    model.lr = lr
    model.l2_reg = 1e-4
    model.beta = beta
    model.train_batch_size = train_batch_size
    model.clipping_value = 100.0
    model.figs_checkpoints_save_freq = figs_checkpoints_save_freq
    model.epoch_cur = start_epoch
    model.warmup_time = warmup_time
    model.beta_final = beta_final
    if start_epoch == 1:
        model.losses_file_exists = False
        model.kls_file_exists = False
        model_utils.book_keeping(model, start_epoch=start_epoch)
    else:
        model.losses_file_exists = True
        model.kls_file_exists = True
        model_utils.book_keeping(model, start_epoch=start_epoch)
        ckpt_path = model.folder_name + '/checkpoints/' + 'ckpt_' + str(start_epoch - 1) + '_weights.path.tar'
        ckpt = torch.load(ckpt_path, map_location=torch.device(model.device))
        model.load_state_dict(ckpt['state_dict'])


def train_motion_epoch(model, train_loader, use_dct=False):
    model.train()
    for i, (motion_seqs) in enumerate(tqdm(train_loader)):

        b_n, f_n, t_n = motion_seqs.shape
        model.b_n, model.f_n, model.t_n = b_n, f_n, t_n

        inputs = motion_seqs.to(model.device).float()
        if use_dct:
            inputs_dct = model_utils.dct(model, inputs)

            #inputs_dct = inputs_dct.reshape(b_n, f_n * t_n)
            mu_hat, logvar_hat, zs, kls = model(inputs_dct.float())
            #mu_hat = mu_hat.reshape(b_n, f_n, t_n)

            inputs_hat = model_utils.dct(model, mu_hat, inverse=True)
        else:
            inputs = inputs.reshape(b_n, f_n * t_n)
            inputs_hat, logvar_hat, zs, kls = model(inputs.float())
        inputs = inputs.reshape(b_n, f_n*t_n)
        inputs_hat = inputs_hat.reshape(b_n, f_n*t_n)
        logvar_hat = logvar_hat.reshape(b_n, f_n*t_n)
        loss = model.cal_loss(inputs, inputs_hat, logvar_hat, kls, 'gaussian')

        model.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.clipping_value)
        model.writer.add_scalar("Gradients/total_gradient_norm", total_norm, model.epoch_cur)
        #if (total_norm < 150) or (model.epoch_cur < 50):
        model.optimizer.step()

    model.beta = model_utils.warmup(model, model.beta, warmup_time=model.warmup_time, beta_final=model.beta_final)

def eval_motion_batch(model, loader, dataset_name='val', use_dct=False):
    with torch.no_grad():
        model.eval()
        for i, (motion_seqs) in enumerate(tqdm(loader)):

            b_n, f_n, t_n = motion_seqs.shape
            model.b_n, model.f_n, model.t_n = b_n, f_n, t_n

            inputs = motion_seqs.to(model.device).float()
            if use_dct:
                inputs_dct = model_utils.dct(model, inputs)

                #inputs_dct = inputs_dct.reshape(b_n, f_n * t_n)
                mu_hat, logvar_hat, zs, kls = model(inputs_dct.float())
                #mu_hat = mu_hat.reshape(b_n, f_n, t_n)

                inputs_hat = model_utils.dct(model, mu_hat, inverse=True)
            else:
                inputs = inputs.reshape(b_n, f_n * t_n)
                inputs_hat, logvar_hat, zs, kls = model(inputs.float())
            inputs = inputs.reshape(b_n, f_n * t_n)
            inputs_hat = inputs_hat.reshape(b_n, f_n * t_n)
            logvar_hat = logvar_hat.reshape(b_n, f_n * t_n)
            loss = model.cal_loss(inputs, inputs_hat, logvar_hat, kls, 'gaussian')

            model_utils.accum_update(model, str(dataset_name)+'_loss', loss)
            model_utils.accum_update(model, str(dataset_name)+'_recon', model.recon_loss)
            if model.variational:
                model_utils.accum_update(model, str(dataset_name)+'_KL', model.KL)
                for key, value in model.KLs.items():
                    model_utils.accum_update(model, str(dataset_name) + '_KL_' + str(key), value)

        model_utils.log_epoch_values(model, dataset_name)


