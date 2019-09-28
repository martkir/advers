import os


def cutout_trial():
    config = {
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',
        'resnet_size': 20,
        'attack': 'cutout',
        'n_holes': 1,
        'length': 16}

    """
    epochs ?
    batch_size ?
    learning_rate?
    optimizer?
    """

    command_string = 'python train.py '
    command_string += ' '.join(['--{} {}'.format(k, v) for k, v in config.items()])
    os.system(command_string)


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


cutout_trial()