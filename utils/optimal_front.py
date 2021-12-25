import pandas as pd
import argparse

parser = argparse.ArgumentParser("Find Optimal Pareto Front for different dataset in NAS_BENCH201 benchmark tool")
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose either cifar10, cifar100 or imagenet')
args = parser.parse_args()

def check_dominance(x, y):
    params_x = df.iloc[x, 2]
    acc_x = df.iloc[x, 1]
    params_y = df.iloc[y, 2]
    acc_y = df.iloc[y, 1]

    if params_x < params_y and acc_x < acc_y:
        return True
    elif params_x == params_y and acc_x < acc_y:
        return True
    elif acc_x == acc_y and params_x < params_y:
        return True
    else:
        return False


def find_optim_front(file):
    df = pd.read_csv(f'fronts/{file}.csv')
    fronts = []
    for x in range(15625):
    # print(x)
    # flag = True
    # s = time.time()
        for y in range(15625):
            # print(y)
            if check_dominance(y,x):
                # flag = False
                break
        fronts.append(x)


def main():
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet':
        non_dominated_front = find_optim_front(args.dataset)
    else:
        raise NameError('Invalid dataset name!')

    with open('f{args.dataset}.txt', 'w') as f:
        for i in non_dominated_front:
            f.write(f'{i}\n')

if __name__ == "__main__":
    main()