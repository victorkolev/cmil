# Conservative Model-based Imitation Learning

## Setup
```
conda create -n cmil python=3.8.11 setuptools=65.5.1 Cython=0.29.32
conda activate cmil
pip install -r requirements.txt
```

## Experiments
To reproduce experiments, follow the steps below: 

```
# Franka Kitchen
python train.py --configs kitchen --logdir logs/kitchen

# ShadowHand Baoding Balls
python train.py --configs shadowhand --logdir logs/shadowhand
```

