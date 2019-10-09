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
    config_normal = {
        'mode': 'normal',
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',
        # model
        'resnet_size': 56,
        # optimizer
        'base_lr': 0.1,
        'wd': 1e-4,
        'momentum': 0.9,
        # train opts:
        'batch_size': 32,
        'epochs': 100,
        # attack opts:
        'attack_name': 'cutout',  # because fastest.
        'n_holes': 1,
        'length': 16
    }

    config_pgd_linf = {
        'mode': 'advers',
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',
        # model
        'resnet_size': 56,  # check.
        # optimizer
        'base_lr': 0.1,  # check.
        'wd': 1e-4,  # check.
        'momentum': 0.9,  # check.
        # train opts:
        'batch_size': 32,  # check.
        'epochs': 100,  # check.
        # attack opts:
        'attack_name': 'pgd_linf',
        'n_iters': 10,
        'epsilon': 32.0}

    config_pgd_l2 = config_pgd_linf
    config_pgd_l2['attack_name'] = 'pgd_l2'
    config_pgd_l2['epsilon'] = 4800

    config_pgd_l1 = config_pgd_linf
    config_pgd_l1['attack_name'] = 'pgd_l1'
    config_pgd_l1['epsilon'] = 612000

    """
    note: in patch gaussian paper on cifar-10 a different model is used. for this experiment i will use the same model
    as in the uar paper.
    """

    # todo: check gauss params.
    # patch_gaussian
    config_patch_gaussian = {
        'mode': 'advers',
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',
        # model
        'resnet_size': 56,
        # optimizer
        'base_lr': 0.1,
        'wd': 1e-4,
        'momentum': 0.9,
        # train opts:
        'batch_size': 32,
        'epochs': 100,
        # attack opts:
        'attack_name': 'patch_gaussian',
        'patch_size': 16,
        'max_scale': 1,
        'sample_up_to': False
    }

    config_list = [
        config_normal,
        config_pgd_linf,
        config_pgd_l2,
        config_pgd_l1,
        config_patch_gaussian
    ]

    run(config_list)


def test_single():
    config = {
        'mode': 'normal',
        'dataset': 'cifar-10',
        'dataset_path': 'data/cifar-10',
        # model
        'resnet_size': 56,
        # optimizer
        'base_lr': 0.1,
        'wd': 1e-4,
        'momentum': 0.9,
        # train opts:
        'batch_size': 32,
        'epochs': 100,
        # attack opts:
        'attack_name': 'cutout',  # because fastest.
        'n_holes': 1,
        'length': 16
    }

    command_string = 'python train.py '
    command_string += ' '.join(['--{} {}'.format(k, v) for k, v in config.items()])
    os.system(command_string)