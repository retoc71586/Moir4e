# Moir4e
2022 Machine Perception project 2 - Human Motion Prediction

To train: bsub -n 6 -W 4:00 -o dct_att_gcn_hype -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --model dct_att_gcn --n_epochs 1000 --lr 0.0005 --use_lr_decay --lr_decay_step 330 --bs_train 128 --bs_eval 128 --nr_dct_dim 64 --loss_ABtype avg_l1 --lr_decay_rate 0.98 --opt adam --kernel_size 40 --clip_gradient --max_norm 1
