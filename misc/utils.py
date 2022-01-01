import numpy as np
import pandas as pd
from nats_bench import create

operations = {
    0 : 'none',
    1 : 'skip_connect',
    2 : 'nor_conv_1x1',
    3 : 'nor_conv_3x3',
    4 : 'avg_pool_3x3',
}

api = create('/content/CS410.M11/NAS_Bench', 'tss', fast_mode=True, verbose=False)

def read_file(file):
    df = pd.read_csv(f'fronts/{file}.csv')
    with open(f'fronts/{file}.txt', 'r') as f:
        file = f.read().splitlines() 
    file = [int(i) for i in file]
    flops = [df.iloc[i,2] for i in file]
    acc = [df.iloc[i,1] for i in file]
    return acc, flops

def query(x, dataset):
    architecture = f'|{operations[x[0]]}~0|+|{operations[x[1]]}~0|{operations[x[2]]}~1|+|{operations[x[3]]}~0|{operations[x[4]]}~1|{operations[x[5]]}~2|'
    idx = api.query_index_by_arch(architecture)
    flops = api.get_cost_info(idx, dataset=dataset, hp='200')['flops']
    error = 100 - api.get_more_info(idx, dataset= dataset, hp='200', is_random= False)['test-accuracy']
    return error, flops