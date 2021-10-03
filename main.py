from datasets.amass import AMASS
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import h5py
import torch.optim as optim

from opt import Options
opt = Options().parse()

import train as train
import models.utils as model_utils

in_n = 50
out_n = 0
input_n= [54, in_n]

print('>>> loading datasets')
train_dataset = AMASS(input_n=in_n, output_n=out_n, split=0)
print('>>> Train dataset length: {:d}'.format(train_dataset.__len__()))
train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=0, pin_memory=False)

val_dataset = AMASS(input_n=in_n, output_n=out_n, split=1)
print('>>> Validation dataset length: {:d}'.format(val_dataset.__len__()))
val_loader = DataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0, pin_memory=False)

#test_dataset = AMASS(input_n=in_n, output_n=out_n, split=2)
#print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
#test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=False)

folder_name="saved_models/"+opt.name
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
else:
    device = "cpu"


import models.VDVAE as nnmodel
model = nnmodel.VDVAE(input_n=input_n, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=opt.p_drop)
#import models.VAE as nnmodel
#model = nnmodel.VAE(input_n=54*50, hidden_layers=[2000, 1000, 500, 100, 50], n_z=50, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=opt.p_drop)
train.initialise(model, start_epoch=opt.start_epoch, folder_name=folder_name, lr=opt.lr, beta=opt.beta, l2_reg=opt.l2_reg, train_batch_size=opt.train_batch_size, warmup_time=opt.warmup_time, beta_final=opt.beta_final)

for epoch in range(opt.start_epoch, opt.n_epochs+1):
    print("Epoch:{}/{}".format(epoch, opt.n_epochs))
    model.epoch_cur = epoch

    use_dct = opt.use_dct
    train.train_motion_epoch(model, train_loader, use_dct=use_dct)
    train.eval_motion_batch(model, train_loader, 'train', use_dct=use_dct)
    train.eval_motion_batch(model, val_loader, 'val', use_dct=use_dct)

    model_utils.save_checkpoint_and_csv(model)
    model.writer.close()