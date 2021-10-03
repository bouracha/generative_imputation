import torch
import sys
import os

import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class AccumValue(object):
    """Object for each variable about which
    we want to accumulate updates

    """
    def __init__(self):
        self.reset()

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

def accum_update(model, key, val):
    """Checks if model is tracking a variable,
    adds it as a variable if not, or updates it if it is

    :param model: the model instance
    :param key: name of variable of which to keep track
    :param val: the value to update the variable
    """
    if key not in model.accum_loss.keys():
        model.accum_loss[key] = AccumValue()
    val = val.cpu().data.numpy()
    model.accum_loss[key].update(val)

def accum_reset(model):
    """Resets all variables being tracked
    for the model instance

    :param model: the model instance
    """
    for key in model.accum_loss.keys():
        model.accum_loss[key].reset()

class CsvLog(object):
    """Maintains arrays for logging values that
    will be saved as csv

    """
    def __init__(self):
        self.reset()

    def update_log(self, head, value):
        self.heads = np.append(self.heads, [head])
        self.values = np.append(self.values, [value])

    def reset(self):
        self.heads = []
        self.values = []


def num_parameters_and_place_on_device(model):
    """PLace model on appropriate device and
    return number of model parameters

    :param model: the model instance
    :return: number of learnable parameters
    """
    #print(model)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(">>> total params: {:.2f}M".format(num_parameters / 1000000.0))
    if model.device == "cuda":
        print("Moving model to GPU")
        model.cuda()
    else:
        print("Using CPU")
    return num_parameters

def define_neurons_layers(n_z_pre, n_z_next, num_layers):
    """

    :param n_z_pre: the layer with the higher number of neurons
    :param n_z_next: the layer with the lower number of neurons
    :param num_layers: number of layers desired in between
    :return: list of layer sizes (list of ints)
    """
    nn_layers = np.linspace(n_z_pre, n_z_next, num_layers+1)
    nn_layers = list(map(int, nn_layers))
    return nn_layers

def warmup(model, cur_beta, warmup_time=200, beta_final=1.0):
    """

    :param epoch: current beta
    :param cur_beta: current beta
    :param warmup_time: number of epochs for which to warm up linearly
    :param beta_final: final beta to which to warm up
    :return: beta after update (if update was required)
    """
    model.writer.add_scalar("Gradients/beta", cur_beta, model.epoch_cur)
    if cur_beta < beta_final:
        cur_beta += beta_final/(1.0*warmup_time)
    if cur_beta >= beta_final:
        cur_beta = beta_final
    return cur_beta

# ===============================================================
#                     VAE specific functions
# ===============================================================

def reparametisation_trick(mu, log_var, device):
    """

    :param mu: The mean of the latent variable to be formed (nbatch, n_z)
    :param log_var: The log variance of the latent variable to be formed (nbatch, n_z)
    :param device: CPU or GPU
    :return: latent variable (nbatch, n_z)
    """
    noise = torch.normal(mean=0, std=1.0, size=log_var.shape).to(torch.device(device))
    z = mu + torch.mul(torch.exp(log_var / 2.0), noise)

    return z

def kullback_leibler_divergence(mu_1, log_var_1, mu_2=None, log_var_2=None, graph=False):
    """

    :param mu: The mean of the latent variable to be formed (nbatch, n_z)
    :param log_var: The log variance of the latent variable to be formed (nbatch, n_z)
    :return: gaussian analytical KL divergence for each datapoint averaged across the
    batch between two gaussian distributions p and q where by default q is N(0,1)
    """
    if mu_2 is None and log_var_2 is None:
        mu_2 = torch.zeros_like(mu_1)
        log_var_2 = torch.zeros_like(log_var_1)
    if graph:
        axis_to_sum = (1, 2)
    else:
        axis_to_sum = 1
    KL_per_datapoint = 0.5 * torch.sum(-1 + log_var_2 - log_var_1 + (torch.exp(log_var_1) + torch.pow((mu_1 - mu_2), 2))/(torch.exp(log_var_2)), axis=axis_to_sum)
    KL = torch.mean(KL_per_datapoint)

    return KL

# ===============================================================
#                     Probabilities
# ===============================================================

def cal_gauss_log_lik(x, mu, log_var=0.0):
    """
    :param x: batch of inputs (bn X fn)
    :return: gaussian log likelihood, and the mean squared error
    """
    MSE = torch.pow((mu - x), 2)
    gauss_log_lik = -0.5*(log_var + np.log(2*np.pi) + (MSE/(1e-8 + torch.exp(log_var))))
    MSE = torch.mean(torch.sum(MSE, axis=1), axis=0)
    gauss_log_lik = torch.mean(torch.sum(gauss_log_lik, axis=1))

    return gauss_log_lik, MSE

def cal_gauss_log_lik_per_datapoint(x, mu, log_var=0.0):
    """
    :param x: batch of inputs (bn X fn)
    :return: gaussian log likelihood, and the mean squared error
    """
    MSE = torch.pow((mu - x), 2)
    gauss_log_lik = -0.5*(log_var + np.log(2*np.pi) + (MSE/(1e-8 + torch.exp(log_var))))
    MSE = torch.sum(MSE, axis=1)
    gauss_log_lik = torch.sum(gauss_log_lik, axis=1)

    return gauss_log_lik, MSE

def cal_bernoulli_log_lik(x, logits):
    """
    :param x: batch of inputs (bn X fn)
    :return: gaussian log likelihood, and the mean squared error (scalar)
    """
    BCE = torch.maximum(logits, torch.zeros_like(logits)) - torch.multiply(logits, x) + torch.log(1 + torch.exp(-torch.abs(logits)))
    BCE_per_sample = torch.sum(BCE, axis=1)
    BCE_avg_for_batch = torch.mean(BCE_per_sample)
    bernoulli_log_lik = -BCE_avg_for_batch

    return bernoulli_log_lik


def cal_VLB(p_log_x, KL, beta=1.0):
    """
    :param x: batch of inputs
    :return: Variational Lower Bound
    """
    VLB = p_log_x - beta*KL

    return VLB

# ===============================================================
#                     Transfoirmations in torch
# ===============================================================

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def dct(model, in_tensor, inverse=False):
    """
    :param model: the model instance
    :param in_tensor: tensor to have it's temproal component DCT'ed assumes
    that the last dim of in_tenosr is the temporal one
    :param inverse: toggle to inverse the transformation (boole)
    """
    t_n = in_tensor.size()[-1]
    dct_matrix, idct_matrix = get_dct_matrix(t_n)
    if not inverse:
        dct_matrix = torch.from_numpy(dct_matrix).to(model.device).float()
        tensor_dct = torch.matmul(in_tensor, dct_matrix)
        return tensor_dct
    else:
        idct_matrix = torch.from_numpy(idct_matrix).to(model.device).float()
        tensor_idct = torch.matmul(in_tensor, idct_matrix)
        return tensor_idct

# ===============================================================
#                     VDVAE bookkeeping functions
# ===============================================================

def book_keeping(model, start_epoch=1):
    """Create bookeeping folders, save model and save
    architectural information and other training conditions

    :param model: the model instance
    :param start_epoch: at what epoch---if training is resuming
    """
    model.accum_loss = dict()

    model.writer = SummaryWriter(log_dir=model.folder_name+'/tensorboard')
    if start_epoch==1:
        os.makedirs(os.path.join(model.folder_name, 'checkpoints'))
        os.makedirs(os.path.join(model.folder_name, 'images'))
        os.makedirs(os.path.join(model.folder_name, 'samples'))
        os.makedirs(os.path.join(model.folder_name, 'poses'))
        write_type='w'
    else:
        write_type = 'a'

    original_stdout = sys.stdout
    with open(str(model.folder_name)+'/'+'architecture.txt', write_type) as f:
        sys.stdout = f
        #if start_epoch==1:
        #    print(model)
        print("Start epoch:{}".format(start_epoch))
        print("Learning rate:{}".format(model.lr))
        print("Training batch size:{}".format(model.train_batch_size))
        print("Clipping value:{}".format(model.clipping_value))
        print("BN:{}".format(model.batch_norm))
        print("l2 Reg (1e-4):{}".format(model.l2_reg))
        print("p_dropout:{}".format(model.p_dropout))
        print("Output variance:{}".format(model.output_variance))
        print("Beta(downweight of KL):{}".format(model.beta))
        print("Activation function:{}".format(model.activation))
        print("Num_parameters:{}".format(model.num_parameters))
        sys.stdout = original_stdout

    model.losses = CsvLog()
    model.kls = CsvLog()



def save_checkpoint_and_csv(model):
    """Save checkpoint and write variables to csvs, also reset logs

    :param model: the model instance
    """
    df = pd.DataFrame(np.expand_dims(model.losses.values, axis=0))
    df_kls = pd.DataFrame(np.expand_dims(model.kls.values, axis=0))
    if model.losses_file_exists:
        with open(model.folder_name + '/' + 'losses.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)
    else:
        df.to_csv(model.folder_name + '/' + 'losses.csv', header=model.losses.heads, index=False)
        model.losses_file_exists = True
    if model.kls_file_exists:
        with open(model.folder_name + '/' + 'kls.csv', 'a') as f:
            df_kls.to_csv(f, header=False, index=False)
    else:
        df_kls.to_csv(model.folder_name + '/' + 'kls.csv', header=model.kls.heads, index=False)
        model.kls_file_exists = True
    state = {'epoch': model.epoch_cur + 1,
             'err': model.accum_loss['val_loss'].avg,
             'state_dict': model.state_dict()}
    if model.epoch_cur % model.figs_checkpoints_save_freq == 0:
        print("Saving checkpoint....")
        file_path = model.folder_name + '/checkpoints/' + 'ckpt_' + str(model.epoch_cur) + '_weights.path.tar'
        torch.save(state, file_path)
    model.losses.reset()
    model.kls.reset()
    accum_reset(model)

def log_epoch_values(model, dataset_name):
    model.losses.update_log('Epoch', model.epoch_cur)
    model.losses.update_log(dataset_name + '_loss', model.accum_loss[str(dataset_name) + '_loss'].avg)
    model.losses.update_log(dataset_name + '_reconstruction', model.accum_loss[str(dataset_name) + '_recon'].avg)
    model.kls.update_log('Epoch', model.epoch_cur)
    model.kls.update_log('Beta', model.beta)
    model.writer.add_scalar(f'loss/loss_' + str(dataset_name), model.accum_loss[str(dataset_name) + '_loss'].avg,
                            model.epoch_cur)
    model.writer.add_scalar(f'loss/reconstructions_' + str(dataset_name),
                            model.accum_loss[str(dataset_name) + '_recon'].avg, model.epoch_cur)
    if model.variational:
        model.losses.update_log(str(dataset_name) + '_KL', model.accum_loss[str(dataset_name) + '_KL'].avg)
        model.writer.add_scalar(f'KLs/total_' + str(dataset_name), model.accum_loss[str(dataset_name) + '_KL'].avg,
                                model.epoch_cur)
        for key, value in model.KLs.items():
            model.kls.update_log(str(dataset_name) + key, model.accum_loss[str(dataset_name) + '_KL_' + str(key)].avg)
            model.writer.add_scalar(f'KLs/' + str(dataset_name) + str(key),
                                    model.accum_loss[str(dataset_name) + '_KL_' + str(key)].avg, model.epoch_cur)