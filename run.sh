#################################################################################
### First run
#################################################################################

#python3 main.py --name HGVAE_warmup200_lre-3 --lr 0.0001 --warmup_time 200 --beta 0.0001 --n_epochs 500 --variational --output_variance --train_batch_size 800 --test_batch_size 800

#VGAE
# python3 main.py --name VGVAE --lr 0.001 --n_epochs 50 --variational --output_variance --train_batch_size 800 --test_batch_size 800
# Vanilla VAE
#python3 main.py --name VAE --lr 0.001 --n_epochs 500 --variational --output_variance --train_batch_size 8212 --test_batch_size 8212 --batch_norm
#python3 main.py --name VAE --lr 0.0001 --start_epoch 131 --n_epochs 500 --variational --output_variance --train_batch_size 8212 --test_batch_size 8212 --batch_norm

#################################################################################
### Occlusions
#################################################################################

#python3 occlusion_experiment.py

######
#python3 main.py --name VDGCVAE_deeep --lr 0.0001 --warmup_time 200 --beta 0.0001 --n_epochs 500 --variational --output_variance --train_batch_size 256 --test_batch_size 256
#python3 main.py --name VDGCVAE_deeep --lr 0.0001 --warmup_time 200 --beta 0.0001 --n_epochs 500 --variational --output_variance --train_batch_size 256 --test_batch_size 256

#################################################################################
### Test runs
#################################################################################

python3 main.py --name "saved_models/HGVAE" --lr 0.0001 --warmup_time 200 --beta 0.0001 --n_epochs 500 --variational --output_variance --train_batch_size 800 --test_batch_size 800