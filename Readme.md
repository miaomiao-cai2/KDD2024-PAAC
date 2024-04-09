# PCCA

## Overview

Official code of "Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias" (2024 KDD ID:848)

## Requirements

- python == 3.9.7

- pytorch == 1.12.0+cu113

- numba == 0.54.1

- numpy == 1.20.0

- faiss-gpu ==1.7.2

- pandas == 1.3.4 

### LightGCN backbone

For models with LightGCN as backbone, For example:

- PCCA  Training:

#### yelp2018

python PAAC_main.py --dataset_name yelp2018 --layers_list '[5]' --cl_rate_list '[10]' --align_reg_list '[1e3]' --lambada_list '[0.8]' --gama_list '[0.8]'

#### gowalla

python PAAC_main.py --dataset_name gowalla --layers_list '[6]' --cl_rate_list '[5]' --align_reg_list '[50]' --lambada_list '[0.2]' --gama_list '[0.2]'
