import numpy as np
from pymoo.core.problem import Problem
from utils import en_decode
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import argparse

parser = argparse.ArgumentParser("NSGA-II algorithm for Bi-Objective Neural Architecture Search Problem")
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose either cifar10, cifar100 or ImageNet16-120')
parser.add_argument('--seed', type=int, help='Random seed for reproducible result')
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='Nums of generation for NSGA-II')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
args = parser.parse_args()


class NAS(Problem):
  def __init__(self, n_var=5, n_obj=2, xl=0, xu=4, dataset='cifar10'):
    super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu , type_var = np.intc)
    self._dataset = dataset
  def _evaluate(self, x, out, *args, **kwargs):
    objs = np.full((x.shape[0], self.n_obj), np.nan)
    for i in range(x.shape[0]):
      _error, _flops = en_decode.query(x[i,:], self._dataset)
      objs[i, 0] = 100 - _error
      objs[i, 1] = _flops
    out["F"] = objs

        
def main():
    np.random.seed(args.seed)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'ImageNet16-120':
        problem = NAS(dataset=args.dataset)
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

if __name__ == "__main__":
    main()
