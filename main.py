import numpy as np
from pymoo.core.problem import Problem
from misc import utils
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_performance_indicator
from pymoo.optimize import minimize
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

all_pops = []

parser = argparse.ArgumentParser("NSGA-II algorithm for Bi-Objective Neural Architecture Search Problem")
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose either cifar10, cifar100 or imagenet')
parser.add_argument('--seed', type=int, default=19522298 ,help='Random seed for reproducible result')
parser.add_argument('--pop_size', type=int, default=200, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=250, help='Nums of generation for NSGA-II')
parser.add_argument('--n_offspring', type=int, default=20, help='number of offspring created per generation')
args = parser.parse_args()


class NAS(Problem):
  def __init__(self, n_var=6, n_obj=2, xl=0, xu=4, dataset='cifar10'):
    super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu , type_var = np.intc)
    self._dataset = dataset
  def _evaluate(self, x, out, *args, **kwargs):
    objs = np.full((x.shape[0], self.n_obj), np.nan)
    for i in range(x.shape[0]):
      _error, _flops = utils.query(x[i,:], self._dataset)
      objs[i, 0] = _error
      objs[i, 1] = _flops
    out["F"] = objs


def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get("F")
    all_pops.append(pop_obj)

def animation(file, all_pops,num_gen=50):
    def update(i):
        plt.title(f'Gen {i}')
        all_pops[generation] = all_pops[generation+i]
        scatter.set_offsets(all_pops[generation])
        return scatter,

    fig = plt.figure(figsize=(15, 8))
    generation = 0
    acc, flops = utils.read_file(file)
    plt.scatter(acc, flops, c='blue', label="Pareto Optimal")
    scatter = plt.scatter(
        all_pops[generation][:, 0], all_pops[generation][:, 1], s=50, c='red')

    anim = FuncAnimation(fig, update, interval=300, frames=num_gen)
    anim.save(f'img\\{file}.gif')

def main():
    np.random.seed(args.seed)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
      problem = NAS(dataset=args.dataset)
    elif args.dataset == 'imagenet':
      problem = NAS(dataset='ImageNet16-120')
    else:
        raise NameError('Invalid dataset name!')
    
    algorithm = NSGA2(
        pop_size=args.pop_size,
        n_offsprings=args.n_offspring,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=0.9, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
        )
    res = minimize(problem,
            algorithm,
            termination=get_termination("n_gen", args.n_gens),
            seed=args.seed,
            callback=do_every_generations,
            save_history=True,
            verbose=True
            )

    with open(f'populations\\{args.dataset}_pop.pkl', 'wb') as f:
      pickle.dump(all_pops, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'populations\\{args.dataset}_res.pkl', 'wb') as f:
      pickle.dump(res.F, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    pf = np.column_stack((utils.read_file(args.dataset)))
    igd = get_performance_indicator("igd", pf)
    print(f"IGD of {args.dataset.upper()}: ", igd.do(res.F))

if __name__ == "__main__":
    main()
