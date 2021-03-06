from datasets.amass import AMASS
from experiments.utils import simulate_motion_occlusions
from experiments.utils import add_noise
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

in_n = 50
out_n = 0

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
else:
    device = "cpu"

import train as train
import models.utils as model_utils
import models.HGVAE as nnmodel
model = nnmodel.HGVAE(input_n=[54, in_n], variational=True, output_variance=True, device=device, batch_norm=False, p_dropout=0.0)
train.initialise(model, start_epoch=501, folder_name="saved_models/HGVAE_warmup200_lre-3", lr=0.001, beta=1.0, l2_reg=1e-4, train_batch_size=800)



print('>>> loading datasets')
test_dataset = AMASS(input_n=in_n, output_n=out_n, split=2)
print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
test_loader = DataLoader(test_dataset, batch_size=400, shuffle=False, num_workers=0, pin_memory=False)

def reconstructions(model, inputs, use_dct=True):
    with torch.no_grad():
        model.eval()
        b_n, f_n, t_n = inputs.shape

        inputs = inputs.to(model.device).float()
        inputs_dct = model_utils.dct(model, inputs)
        #inputs_dct = inputs_dct.reshape(b_n, f_n * t_n)
        mu_hat, logvar_hat, zs, kls = model(inputs_dct.float(), latent_resolution=999)
        #mu_hat = mu_hat.reshape(b_n, f_n, t_n)
        inputs_hat = model_utils.dct(model, mu_hat, inverse=True)

    return inputs_hat.detach()

def test_degraded_mse(test_loader, num_occlusions=0):
    MSE_log = []
    MSE_recon_log = []
    for i, (motion_gt) in enumerate(tqdm(test_loader)):
        batch_size, f_n, t_n = motion_gt.shape

        ### Simulate degradation
        X_occluded, occlusion_mask = simulate_motion_occlusions(motion_gt, num_occlusions=num_occlusions)
        #X_occluded = add_noise(motion_gt, alpha=num_occlusions)
        motion_occluded = torch.from_numpy(X_occluded).to(device).float()
        occlusion_mask = torch.from_numpy(occlusion_mask).to(device).float()
        motion_gt = motion_gt.to(device).float()

        motion_gt_dct = model_utils.dct(model, motion_gt)
        posterior_gt = model.cal_posterior(motion_gt_dct).detach()

        #recons_of_occluded = reconstructions(model, motion_occluded, use_dct=True)
        map_imputation, (posterior_init, posterior_final) = model.gradient_ascent_on_posterior(motion_occluded, occlusion_mask)

        MSE_per_datapoint = torch.sum(torch.pow((motion_gt-motion_occluded), 2), axis=(1,2))
        MSE_MAP_per_datapoint = torch.sum(torch.pow((motion_gt-map_imputation ), 2), axis=(1,2))

        head=['posterior_init','posterior_final','posterior_gt','mse_occluded','mse_map']
        df = pd.DataFrame(posterior_init.cpu())
        #print(df.shape)
        #df["posterior_init"] = posterior_init.cpu()
        df["posterior_final"] = posterior_final.cpu()
        df["posterior_gt"] = posterior_gt.cpu()
        df["MSE_occluded"] = MSE_per_datapoint.cpu()
        df["MSE_MAP_occluded"] = MSE_MAP_per_datapoint.cpu()
        #print(df.shape)
        #print(i)
        if i==0:
            df.to_csv("saved_models/HGVAE_warmup200_lre-3/10_optim" + '/' + str(num_occlusions) + '_occlusions_inputs_MAP.csv', header=head, index=False)
        else:
            with open("saved_models/HGVAE_warmup200_lre-3/10_optim" + '/' + str(num_occlusions) + '_occlusions_inputs_MAP.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)

        MSE = torch.mean(MSE_per_datapoint, axis=0)
        MSE_recon = torch.mean(MSE_MAP_per_datapoint, axis=0)
        MSE_log.append(MSE.cpu().numpy())
        MSE_recon_log.append(MSE_recon.cpu().numpy())
    return np.mean(MSE_log), np.mean(MSE_recon_log)



for num_occlusions in [0, 1, 3, 10, 30, 100, 300, 1000, 2700]:
    MSE_log = []
    MSE_recon_log = []
    for i in range(0, 1):
        MSE, MSE_recon = test_degraded_mse(test_loader, num_occlusions=num_occlusions)
        MSE_log.append(MSE)
        MSE_recon_log.append(MSE_recon)
    print('testing MSE: {} +- {} for {} occlusions'.format(np.mean(MSE_log), np.std(MSE_log), num_occlusions))
    print('testing MSE Recon: {} +- {} for {} occlusions'.format(np.mean(MSE_recon_log), np.std(MSE_recon_log), num_occlusions))

