# NTU Fintech 2023 HW1: Robotic Advisor

## Overview
In this homework, you need to design a trading strategy to maximize the return of stock trading over a future period of time.

## Getting Started

**Conda Environment**: 
1. Create a new Conda environment: 
```conda create -n fintech1 python=3.9```

2. Activate the Conda environment:
``` conda activate fintech1 ```

3. Installing packeges:
```
conda config --append channels conda-forge 
conda install numpy=1.23.4 pandas=2.1.1 scipy=1.11.3 cython=3.0.2
conda install -c conda-forge ta-lib==0.4.19
pip install tqdm scikit-learn==1.3.1 torch==2.0.1 stable-baselines3==2.1.0
```

**Train a Robot**
1. Type the command python train.py batch_size buffer_size gamma.
A good set of parameters is batch_size=256, buffer_size=20000 and gamma=0.95.
2. To estimate the overall performance, just type the command:
```python rrEstimate.py 0050.TW-short.csv```


