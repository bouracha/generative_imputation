# Generative Imputation


## Dependencies
Some older versions may work. But we used the following:

* cuda 10.1 (batch size depends on GPU memory)
* Python 3.6.9
* [Pytorch](https://github.com/pytorch/pytorch) 1.6.0
* [progress 1.5](https://pypi.org/project/progress/)
* Tensorboard

## Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[AMASS](https://amass.is.tue.mpg.de/index.html) was obtained from the [repo](https://amass.is.tue.mpg.de/download.php), you need to make an account.

Once downloaded the datasets should be added to the datasets folder, example below.

![Example](datasets/data_structure_example.png "Example of how datasets folder should look")

It is necessary to also add a saved_models folder as each trained model will produce a lot of checkpoints and data as it trains. If training several models it is cleaner to have a separate folder for each of these sub-folders, so saving checkpoints to folders within a saved_models folder is hardcoded.

## Training commands
To train HG-VAE as in the paper:
```bash
python3 main.py --name "HGVAE" --lr 0.0001 --warmup_time 200 --beta 0.0001 --n_epochs 500 --variational --output_variance --train_batch_size 800 --test_batch_size 800
```
see opt.py for all training options. By default checkpoints are saved every 10 epochs. Training may be stop, and resumed by using --start_epoch flag, for example
```bash
python3 main.py --start_epoch 31 --name "HGVAE" --lr 0.0001 --warmup_time 200 --beta 0.0001 --n_epochs 500 --variational --output_variance --train_batch_size 800 --test_batch_size 800
```
will start retraining from the checkpoint saved after epoch 30. We also use the start_epoch flag to select the checkpoint to use when using the trained model.

## Licence

MIT

## Paper

If you use our code, please cite:

```
@article{DBLP:journals/corr/abs-2111-12602,
  author    = {Anthony Bourached and
               Robert Gray and
               Ryan{-}Rhys Griffiths and
               Ashwani Jha and
               Parashkev Nachev},
  title     = {Hierarchical Graph-Convolutional Variational AutoEncoding for Generative
               Modelling of Human Motion},
  journal   = {CoRR},
  volume    = {abs/2111.12602},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.12602},
  eprinttype = {arXiv},
  eprint    = {2111.12602},
  timestamp = {Fri, 26 Nov 2021 13:48:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-12602.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
