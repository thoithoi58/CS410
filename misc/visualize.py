import sys
# update your projecty root path before running
sys.path.insert(0, 'C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import argparse
from misc import utils


parser = argparse.ArgumentParser("NSGA-II algorithm for Bi-Objective Neural Architecture Search Problem")
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose either cifar10, cifar100 or ImageNet16-120')
parser.add_argument('--type', type=str, default='graph' ,help='Visualize type, either "graph" or "animation" or "both"')
parser.add_argument('--n_gens', type=int, default=250, help='Only for animation, nums of generation to visualize')
args = parser.parse_args()

def read_pickle(file):
    with open(f'populations\\{file}_pop.pkl', 'rb') as f:
        pop = pickle.load(f)
    with open(f'populations\\{file}_res.pkl', 'rb') as f:
        res = pickle.load(f)
    return pop, res


def visualize(dataset, res, all_pops):
  acc, flops = utils.read_file(dataset)
  plt.rcParams.update({'font.size': 23})
  plt.rcParams["figure.figsize"] = (15,8)
  fig = plt.figure()
  plt.xlabel("Error rate")
  plt.ylabel("FLOPS")
  plt.scatter(res[:,0], res[:,1], c='red', label='Last generation')
  plt.scatter(all_pops[0][:,0], all_pops[0][:,1], c='green', label='First generation')
  plt.scatter(acc, flops, c='blue', label="Pareto Optimal front")
  plt.legend()
  fig.savefig(f'img\\{dataset}.png')


def animation(dataset, all_pops,num_gen=50):
    def update(i):
        plt.title(f'Gen {i}')
        all_pops[generation] = all_pops[generation+i]
        scatter.set_offsets(all_pops[generation])
        return scatter,

    acc, flops = utils.read_file(dataset)
    fig = plt.figure(figsize=(15, 8))
    generation = 0
    plt.scatter(acc, flops, c='blue', label="Pareto Optimal")
    scatter = plt.scatter(
        all_pops[generation][:, 0], all_pops[generation][:, 1], s=50, c='red')

    anim = FuncAnimation(fig, update, interval=300, frames=num_gen)
    anim.save(f'img\\{dataset}.gif')

def main():
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet':
        pass
    else:
        raise NameError('Invalid dataset!')
    if args.type == 'graph':
        all_pops, res = read_pickle(args.dataset)
        visualize(args.dataset, res, all_pops)
    elif args.type == 'animation':
        all_pops, res = read_pickle(args.dataset)
        animation(args.dataset, all_pops, args.n_gens)
    elif args.type == 'both':
        all_pops, res = read_pickle(args.dataset)
        visualize(args.dataset, res, all_pops)
        animation(args.dataset, all_pops, args.n_gens)
    else:
        raise NameError('Invalid visualize type!')

if __name__ == "__main__":
    main()
