import shutil
import time
import datetime
import os
import uuid
import subprocess


def run(config_list):
    """ Run multiple experiments at once.

    Each experiment is specified by a config dictionary. Each config is turned into a .sh script to be executed. Before
    the .sh scripts are created all old scripts are deleted.

    Args:
        config_list: list of config dictionaries.
    """

    # delete all old .sh scripts:
    for item in os.listdir('scripts'):
        if item.endswith(".sh"):
            os.remove(os.path.join('scripts', item))

    # create the commands:
    command_string_list = []
    for config in config_list:
        command_string = 'python train.py '
        command_string += ' '.join(['--{} {}'.format(k, v) for k, v in config.items()])
        command_string_list.append(command_string)

    # create the .sh files:
    for i, command_string in enumerate(command_string_list):
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]
        sh_filename = 'scripts/{}-{}-{}.sh'.format('command_{}'.format(i), time_str, run_id)

        with open(sh_filename, 'w+') as f:
            shutil.copy2('scripts/template.sh', sh_filename)

        with open(sh_filename, 'a') as f:
            f.write('\n')
            f.write(command_string)

        time.sleep(0.2)

    os.chdir('scripts')
    list_of_scripts = [item for item in
                       subprocess.run(['ls'], stdout=subprocess.PIPE).stdout.split(b'\n') if
                       item.decode('utf-8').endswith('.sh')]

    for job in list_of_scripts:
        str_to_run = 'sbatch {}'.format(job.decode("utf-8"))
        print(str_to_run)
        os.system(str_to_run)
        time.sleep(3)

    os.chdir('..')


def experiment_1():
    config_cutout = {
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',  # where cifar-10 is downloaded (if not already there).
        'pre_augment': True,  # which trainer to use.
        'adv_train': True,  # whether to adv. train.
        'preprocess_options': 'advers-standard-normalize',
        'resnet_size': 20,
        'checkpoint_dir': 'train',
        'base_lr': 0.1,
        'wd': 1e-4,
        'momentum': 0.9,
        'batch_size': 128,  # default 32.
        'epochs': 100,
        'attack_name': 'cutout',
        'n_holes': 1,  # num. patches.
        'length': 16,  # patch_size.
    }  # default 90.

    config_patch_gauss = {
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',
        'pre_augment': True,
        'adv_train': True,
        'preprocess_options': 'advers-standard-normalize',
        'resnet_size': 20,
        'checkpoint_dir': 'train',
        'base_lr': 0.1,
        'wd': 1e-4,
        'momentum': 0.9,
        'batch_size': 128,
        'epochs': 100,
        'attack_name': 'patch_gaussian',
        'patch_size': 16,
        'max_scale': 1,
        'sample_up_to': False}

    config_list = [config_patch_gauss, config_cutout]
    run(config_list)

    # todo: IMPORTANT: Fix the cehckpoint location - wrong atm!





    pass


def cutout_trial():
    """
    Patch Gaussian Implementation details:

    Patch Gaussian is trained on adversarial examples only. The data augmentations are applied in the
    pre-processing stage (not during training). The pre-processing steps are:
    1. Apply Patch Gaussian on unnormalized images in [0, 255] range.
    2. Apply Random Flipping and Cropping.
    3. Normalize.

    The above preprocessing steps are specified using the --preprocess_options flag. More specifically, by writing:
    <--preprocess_options advers standard normalize>.

    <--pre_augment True> specifies that the adversarial augmentations are applied at the preprocessing stage.
    """

    config = {
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',  # where cifar-10 is downloaded (if not already there).
        'pre_augment': True,  # which trainer to use.
        'adv_train': True,  # whether to adv. train.
        'preprocess_options': 'advers-standard-normalize',
        'resnet_size': 20,
        'checkpoint_dir': 'train',
        'base_lr': 0.1,
        'wd': 1e-4,
        'momentum': 0.9,
        'batch_size': 128,  # default 32.
        'epochs': 100,
        'attack_name': 'cutout',
        'n_holes': 1,  # num. patches.
        'length': 16,  # patch_size.
        }  # default 90.

    command_string = 'python train.py '
    command_string += ' '.join(['--{} {}'.format(k, v) for k, v in config.items()])
    os.system(command_string)