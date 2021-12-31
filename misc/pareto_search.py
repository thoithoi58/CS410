import sys
# update your projecty root path before running
sys.path.insert(0, 'C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project')

import pandas as pd
import argparse

parser = argparse.ArgumentParser("Find Optimal Pareto Front for different dataset in NAS_BENCH201 benchmark tool")
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose either cifar10, cifar100 or imagenet')
args = parser.parse_args()

def check_dominance(x, y, df):
    flops_x = df.iloc[x, 2]
    error_x = df.iloc[x, 1]
    flops_y = df.iloc[y, 2]
    error_y = df.iloc[y, 1]

    if flops_x < flops_y and error_x < error_y:
        return True
    elif flops_x == flops_y and error_x < error_y:
        return True
    elif error_x == error_y and flops_x < flops_y:
        return True
    else:
        return False

def find_optim_front(file):
    df = pd.read_csv(f'optimal_front/{file}.csv')
    fronts = []
    for x in range(15625):
        flag = True
        for y in range(15625):
            if check_dominance(y, x, df):
                flag = False
                break
        if flag:
            fronts.append(x)
            print(fronts[-1])
    return fronts

def main():
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet':
        non_dominated_front = find_optim_front(args.dataset)
    else:
        raise NameError('Invalid dataset name!')

    with open(f'optimal_front\\{args.dataset}.txt', 'a+') as f:
        for i in non_dominated_front:
            f.write(f'{i}\n')

if __name__ == "__main__":
    main()