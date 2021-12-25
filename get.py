# from utils import en_decode
import pandas as pd
from nats_bench import create

api = create('C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project\\NAS_Bench', 'tss', fast_mode=True)

with open('imagenet.csv', 'a') as f:
    for i in range(15625):
        print(i)
        params = api.get_cost_info(i, dataset='ImageNet16-120', hp='200')['params']
        acc = 100. - api.get_more_info(i, dataset= 'ImageNet16-120',hp='200', is_random= False)['test-accuracy']
        f.write(f'{i}, {acc}, {params}\n')