import numpy as np
import logging
from pymoo.core.problem import Problem
from utils import en_decode
from nats_bench import create
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self):
        super().__init__(n_var=5, n_obj=2, xl=0, xu=4 , type_var = int)
        self._n_evaluated = 0  # keep track of how many architectures are sampled

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))

            # call back-propagation training
            
            genome = en_decode.decoding(x[i, :])
            performance, flops = en_decode.query(genome)

            self._n_evaluated += 1

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 - performance
            objs[i, 1] = flops

        out['F'] = objs

        

def main():
    problem = NAS()
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )
    res = minimize(problem,
               algorithm,
               termination=get_termination("n_gen", 40),
               seed=1,
               save_history=True
               )

    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)


if __name__ == "__main__":
    main()
