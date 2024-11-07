import argparse
from itertools import product


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def generate_all_binary_vectors(n: int):
    return [list(p) for p in product([0, 1], repeat=n)]

# import time
# start_time = time.time()
# generate_all_binary_vectors(50)
# print("--- %s seconds ---" % (time.time() - start_time))