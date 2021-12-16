import numpy as np
from nats_bench import create

operations = {
    0 : 'none',
    1 : 'skip_connect',
    2 : 'nor_conv_1x1',
    3 : 'nor_conv_3x3',
    4 : 'avg_pool_3x3',
}

api = create('C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project\\NAS_Bench', 'tss', fast_mode=True)

def decoding(x):
    tmp = []
    x = np.array(x).astype(int)
    for i in x:
        tmp.append(operations[i])
    # print(tmp)
    return np.array(tmp)
        
def query(x):
    arch_test = f'|{x[0]}~0|+|{x[1]}~0|{x[1]}~1|+|{x[2]}~0|{x[2]}~1|{x[3]}~2|'
    ind = api.query_index_by_arch(arch_test)
    cost = api.get_cost_info(ind, dataset='cifar100', hp='12')
    info = api.get_more_info(ind, dataset= 'cifar100',hp='12', is_random= False)
    test_acc = info['test-accuracy']
    flops = cost['flops']
    return test_acc, flops