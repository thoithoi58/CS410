# CS410.M11 - Neural networks and Genetic Algorithm
This is our implementation for Bi-Objective Optimization for Neural Architecture Search Problem

<p align="center">
  <img src="https://github.com/thoithoi58/CS410.M11/blob/master/img/nsga2.png" />
</p>

## Requirements
``` 
Python >= 3.6.8, PyTorch >= 1.5.1, torchvision >= 0.6.1, pymoo == 0.5.0
```
## Architecture search
To run architecture search:
``` shell
python main.py --dataset cifar10 --pop_size 40 --n_gen 50 --n_offspring 40
```
