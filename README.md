# CS410.M11 - Neural networks and Genetic Algorithm
This is our implementation for Multi-Objective Optimization for Neural Architecture Search Problem

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
python main.py --dataset cifar10 --pop_size 200 --n_gens 250 --n_offspring 20
```
## Visualization
To visualize the architectures:
``` shell
python misc/visualize.py --dataset cifar10 --type graph           
```
Remember to update your projecty root path before running

## Results
CIFAR-10                   |  CIFAR-100                | ImageNet-16-120
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/thoithoi58/CS410.M11/blob/master/img/cifar10.gif)  |  ![](https://github.com/thoithoi58/CS410.M11/blob/master/img/cifar100.gif)  |  ![](https://github.com/thoithoi58/CS410.M11/blob/master/img/imagenet.gif)

