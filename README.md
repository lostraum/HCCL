# Hierarchical Cross-view Contrastive Learning for Heterogeneous Graph Neural Network
## HCCL
## Environment Settings
> python==3.8.10 \
> scipy==1.6.0 \
> torch==1.11.0 \
> numpy==1.21.5 \
> scikit_learn==1.1.2

GPU: RTX 3090(24GB) 

CPU: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz 

## Usage

Go into ./code, and then you can use the following commend to run our model:
> python main.py acm --gpu=0

Here, "acm" can be replaced by "dblp", "aminer" or "freebase".

##  Credit
```
The development of the code is based on https://github.com/liun-online/HeCo
```

