from nats_bench import create

api = create('C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project\\NAS_Bench', 'tss', fast_mode=True)

def query(x):
    print(x)
    # arch = f'|{x[0]}~0|+|{x[0]}~0|{x[1]}~1|+|{x[2]}~0|{x[3]}~1|x[4]~2|'
    # idx = api.query_index_by_arch(arch)
    # info = api.get_more_info(idx, dataset='cifar100', hp='12', is_random=False)
    # cost = api.get_cost_info(idx, dataset='cifar100', hp='12')
    # return info['test-accuracy'], cost['flops']