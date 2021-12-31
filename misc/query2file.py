import sys
# update your projecty root path before running
sys.path.insert(0, 'C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project')

from nats_bench import create
import argparse

parser = argparse.ArgumentParser("Save NasBench201 architectures info to csv file for faster query.")
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose either cifar10, cifar100 or imagenet')
args = parser.parse_args()


api = create('C:\\Users\\owcap\\Documents\\Learning\\CS410\\Final Project\\NAS_Bench', 'tss', fast_mode=True)

def save(file):
    if file == 'cifar10' or file == 'cifar100':
        dataset = file
    else:
        dataset = 'ImageNet16-120'
    with open(f'fronts\\{file}.csv', 'a') as f:
        f.write('idx,error,flop\n')
        for i in range(15625):
            # print(i)
            flops = api.get_cost_info(i, dataset=dataset, hp='200')['flops']
            error = 100. - api.get_more_info(i, dataset= dataset,hp='200', is_random= False)['test-accuracy']
            f.write(f'{i}, {error}, {flops}\n')



def main():
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet':
        save(args.dataset)
    else:
        raise NameError('Invalid dataset name!')

if __name__ == '__main__':
    main()