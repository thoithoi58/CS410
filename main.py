import numpy as np
from pymoo.core.problem import Problem
from misc import utils
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_performance_indicator
from pymoo.optimize import minimize
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("NSGA-II algorithm for Bi-Objective Neural Architecture Search Problem")
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose either cifar10, cifar100 or ImageNet16-120')
parser.add_argument('--seed', type=int, help='Random seed for reproducible result')
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='Nums of generation for NSGA-II')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
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

def visualize(dataset, solution):
  acc, flops = utils.read_file(dataset)
  plt.rcParams["figure.figsize"] = (15,8)
  fig, ax = plt.subplots()
  plt.xlabel("Error rate")
  plt.ylabel("FLOPS")
  ax.scatter(solution.F[:,0], solution.F[:,1], c='red', label='NSGA-II')
  ax.scatter(acc, flops, c='blue', label="Pareto Optimal")
  ax.legend()
  plt.figtext(0.5, 0.01, dataset, wrap=True, horizontalalignment='center', fontsize=12)
  plt.savefig(f'img\\{dataset}.png')

        
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
        crossover=get_crossover("int_sbx", prob=0.2, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
        )
    res = minimize(problem,
            algorithm,
            termination=get_termination("n_gen", args.n_gens),
            seed=1,
            save_history=True,
            verbose=True
            )
    visualize(args.dataset, res)
    pf = np.column_stack((utils.read_file(args.dataset)))
    print(pf)
    igd = get_performance_indicator("igd", pf)
    print(f"IGD of {args.dataset.upper()}: ", igd.do(res.F))

if __name__ == "__main__":
    main()
