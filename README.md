# Moir4e
2022 Machine Perception project 2 - Human Motion Prediction

You can find all info regarding the architecture in the project report. 
The dct_att_gcn model scored 1st on the public leaderboard achieving state of the art performance. 

To train dct att gcn model on cluster: 
``` $ bsub -n 6 -W 4:00 -o model_dct_att_gcn -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py ```

To instead train the original lstm try run:
``` python train.py --model seq2seq_lstm --n_epochs 1000 --lr 0.0005 --use_lr_decay --lr_decay_step 330 --bs_train 128 --bs_eval 128 --nr_dct_dim 64 --lr_decay_rate 0.98 --opt adam --kernel_size 40 --clip_gradient --max_norm 1 ```
