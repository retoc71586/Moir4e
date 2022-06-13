# Moir4e
2022 Machine Perception project 2 - Human Motion Prediction

To train on cluster: 

``` $ bsub -n 6 -W 4:00 -o dct_att_gcn_hype -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py ```