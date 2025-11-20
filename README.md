# FedReLa

## Environment
```bash
conda create -n FedReLa python=3.10 -y
conda activate FedReLa
pip install -r requirements.txt
```
PyTorch 2.1.2 + CUDA 12.2 is pinned. If you are on a different GPU stack, install the wheel that matches your driver.

## Dataset Preparation
No manual preprocessing is needed. Both CIFAR-10 and CIFAR-100 are downloaded via TorchVision into `./cifar_lt/` at runtime. If you prefer to keep the raw tarballs in another location, set the `TORCH_HOME` or `XDG_CACHE_HOME` environment variable before launching the jobs.

### CIFAR-10 / FedLoGE
```bash
python fedloge.py --dataset cifar10 --model resnet18 --alpha_dirichlet 0.1 --IF 0.02 \
  --beta 0 --gpu 0 --num_users 40 --frac 1 --ghead g_head \
  --seed 1 --thre 5 --relabel_cal 300 --relabel_start 500 --rounds 550 \
  --id cifar10_exp_dir01_if50_relabel_thre5p_1
python realignment.py --dataset cifar10 --model resnet18 --alpha_dirichlet 0.1 --IF 0.02 \
  --beta 0 --gpu 0 --num_users 40 --frac 1 --ghead g_head \
  --seed 1 --thre 5 --relabel_cal 300 --relabel_start 500 --rounds 550 \
  --relabeltest True --id cifar10_exp_dir01_if50_relabel_thre5p_1
python realignment.py --dataset cifar10 --model resnet18 --alpha_dirichlet 0.1 --IF 0.02 \
  --beta 0 --gpu 0 --num_users 40 --frac 1 --ghead g_head \
  --seed 1 --thre 5 --relabel_cal 300 --relabel_start 500 --rounds 550 \
  --id cifar10_exp_dir01_if50_relabel_thre5p_1
```


### CIFAR-10 / Fed-ETF
```bash
python fed_etf.py --dataset cifar10 --model resnet18 --alpha_dirichlet 0.1 --IF 0.02 \
  --beta 0 --gpu 0 --num_users 40 --frac 1 --ghead g_head \
  --seed 1 --thre 20 --relabel_cal 300 --relabel_start 500 --rounds 550 \
  --id etf_cifar10_exp_dir01_if50_relabel_thre20p_1
```

### CIFAR-100 / FedLoGE
```bash
python fedloge_100.py --dataset cifar100 --model resnet34 --alpha_dirichlet 0.1 --IF 0.02 \
  --beta 0 --gpu 0 --num_users 10 --frac 1 --ghead g_head \
  --seed 1 --thre 20 --relabel_cal 300 --relabel_start 500 --rounds 550 \
  --id cifar100_exp_dir01_if50_relabel_thre20p_1
```

### CIFAR-100 / Fed-ETF
```bash
python fedetf_100.py --dataset cifar100 --model resnet34 --alpha_dirichlet 0.1 --IF 0.02 \
  --beta 0 --gpu 0 --num_users 10 --frac 1 --ghead g_head \
  --seed 1 --thre 20 --relabel_cal 300 --relabel_start 500 --rounds 550 \
  --id etf_cifar100_exp_dir01_if50_relabel_thre20p_1
```


