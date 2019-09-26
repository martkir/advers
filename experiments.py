import os


def train_example():

    config = {
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',
        'resnet_size': 56,
        'attack': 'elastic',
        'epsilon': 8.0,
        'n_iters': 30,
        'step_size': 1}

    command_string = 'python train.py '
    command_string += ' '.join(['--{} {}'.format(k, v) for k, v in config.items()])
    os.system(command_string)

train_example()