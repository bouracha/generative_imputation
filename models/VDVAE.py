#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils

from models.encoders import GraphVDEncoder
from models.decoders import GraphVDDecoder

from torch.nn.parameter import Parameter
import math

from models.layers import FullyConnected
from models.layers import ParametiseInputs


class VDVAE(nn.Module):
    def __init__(self, input_n=[54, 50], act_fn=nn.GELU(), variational=False, output_variance=False, device="cuda", batch_norm=False, p_dropout=0.0):
        """
        :param input_n: num of input feature; (graph_n, feature_n)
        :param hidden_layers: num of hidden feature, decoder is made symmetric
        :param n_z: latent variable size
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VDVAE, self).__init__()
        print(">>> creating model")

        self.activation = act_fn
        self.variational = variational
        self.output_variance = output_variance
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout

        self.encoder = GraphVDEncoder(input_n=input_n, act_fn=self.activation, device=self.device, batch_norm=self.batch_norm, p_dropout=self.p_dropout)
        self.decoder = GraphVDDecoder(input_n=input_n, encoder_activation_sizes=self.encoder.level_output_sizes, act_fn=self.activation, device=self.device, batch_norm=self.batch_norm, p_dropout=self.p_dropout)

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x, z_0=None, one_hot_labels=None, latent_resolution=999):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        #Bottom Up
        encoder_activations = self.encoder.forward(x)

        # Top Down
        self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs = self.decoder(encoder_activations=encoder_activations, one_hot_labels=one_hot_labels, latent_resolution=latent_resolution)


        return self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs



    def generate(self, z, distribution='gaussian', latent_resolution=0, z_prev_level=0, one_hot_labels=None):
        """
        :param z: batch of random variables
        :return: batch of generated samples
        """
        if z_prev_level==0 and z!=None:
            self.decoder.zs["0"] = z
            self.decoder.residuals_dict["0"] = self.decoder.resize_conv_0(self.decoder.zs["0"])

        for i in range(z_prev_level, len(self.decoder.encoder_activation_sizes)-2):
            self.decoder.KLs[str(i+1)], self.decoder.zs[str(i+1)] = self.decoder.top_down_decode(level=i, X_supplied=False, latent_resolution=latent_resolution)

        decoder_output_final = self.decoder.decoder_block_final(self.decoder.residuals_dict[str(len(self.decoder.encoder_activation_sizes)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.decoder.reparametisation_output(decoder_output_final)

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        if distribution=='gaussian':
            return self.reconstructions_mu
        elif distribution=='bernoulli':
            return self.bernoulli_output


    def cal_loss(self, x, mu_hat, logvar_hat=None, KLs=None, distribution='gaussian'):
        """
        :param x: batch of inputs
        :return: loss of reconstructions
        """

        if distribution=='gaussian':
            if not self.output_variance:
                b_n, f_n = mu_hat.shape
                logvar_hat = torch.ones((b_n, f_n)).to(self.device).float()
            self.log_lik, self.mse = utils.cal_gauss_log_lik(x, mu_hat, logvar_hat)
            self.recon_loss = self.mse
        elif distribution=='bernoulli':
            self.log_lik = utils.cal_bernoulli_log_lik(x, mu_hat)
            self.recon_loss = -self.log_lik

        if self.variational:
            self.KL = sum(KLs.values())
            self.VLB = utils.cal_VLB(self.log_lik, self.KL, self.beta)
            self.loss = -self.VLB
        else:
            self.loss = -self.log_lik
        return self.loss

    def cal_posterior(self, x, latent_resolution=999):
        #Bottom Up
        b_n = x.shape[0]
        encoder_activations = self.encoder.forward(x)

        # Top Down
        self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs = self.decoder(encoder_activations=encoder_activations, latent_resolution=latent_resolution)

        self.z_posterior_mus, self.z_posterior_log_vars = self.decoder.z_posterior_mus, self.decoder.z_posterior_log_vars
        self.z_prior_mus, self.z_prior_log_vars = self.decoder.z_prior_mus, self.decoder.z_prior_log_vars

        self.posterior = torch.zeros(b_n).to(self.device)
        self.posteriors = {}
        for level in range(0, len(self.decoder.encoder_activation_sizes)-2):
            y = self.z_posterior_mus[str(level)]
            mu = self.z_prior_mus[str(level)]
            log_var = self.z_prior_log_vars[str(level)]

            y = y.reshape(b_n, -1)
            mu = mu.reshape(b_n, -1)
            log_var = log_var.reshape(b_n, -1)
            self.posteriors[str(level)], _ = utils.cal_gauss_log_lik_per_datapoint(y, mu, log_var=log_var)
            self.posterior = self.posterior + self.posteriors[str(level)] #log probabilities!

        return self.posterior


    def gradient_ascent_on_posterior(self, inputs_occluded, occlusion_mask):
        batch_size, f_n, t_n = inputs_occluded.shape
        self.eval()

        parametise_inputs = ParametiseInputs(inputs_occluded)
        parametise_inputs.to(self.device)

        for param in self.parameters():
            param.requires_grad = False
        parametise_inputs.inputs_as_parameters.requires_grad = True
        # optimizer = torch.optim.SGD({parametise_inputs.inputs_as_parameters}, lr=1e4)
        optimizer = torch.optim.Adam({parametise_inputs.inputs_as_parameters}, lr=1.0)

        posterior_max = -1e10 * torch.ones(batch_size).to(self.device)
        inputs_best = torch.zeros(inputs_occluded.shape).to(self.device)
        for i in range(10):
            inputs_with_parameters = parametise_inputs(inputs_occluded, occlusion_mask)
            inputs_with_parameters_dct = utils.dct(self, inputs_with_parameters)
            posterior = self.cal_posterior(inputs_with_parameters_dct.float(), latent_resolution=999)
            if i==0:
                posterior_init = posterior

            self.optimizer.zero_grad()
            neg_log_posterior = -torch.mean(posterior)
            neg_log_posterior.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            optimizer.step()

            best_posteriors_bool = (posterior.detach() > posterior_max.detach())
            posterior_max[best_posteriors_bool] = posterior[best_posteriors_bool].detach()
            inputs_best[best_posteriors_bool, :, :] = inputs_with_parameters[best_posteriors_bool, :, :].detach()

        return inputs_best.detach(), (posterior_init.detach(), posterior_max.detach())









