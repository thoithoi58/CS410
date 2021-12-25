import numpy as np
from nats_bench import create

operations = {
    0 : 'none',
    1 : 'skip_connect',
    2 : 'nor_conv_1x1',
    3 : 'nor_conv_3x3',
    4 : 'avg_pool_3x3',
}

api = create('C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project\\NAS_Bench', 'tss', fast_mode=True, verbose=False)

def query(x, dataset):
    architecture = f'|{operations[x[0]]}~0|+|{operations[x[1]]}~0|{operations[x[2]]}~1|+|{operations[x[3]]}~0|{operations[x[4]]}~1|{operations[x[5]]}~2|'
    idx = api.query_index_by_arch(architecture)
    cost = api.get_cost_info(idx, dataset='cifar10', hp='200')
    info = api.get_more_info(idx, dataset= 'cifar10',hp='200', is_random= False)
    test_acc = info['test-accuracy']
    flops = cost['params']
    return test_acc, flops


