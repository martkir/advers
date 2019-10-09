import click
from trainer import CIFAR10Experiment
from common import pyt_common
from common.flags_holder import FlagHolder
from attacks import Cutout, PatchGaussian, PGDAttack
from common.logger import Logger
import datetime
import os
import uuid


EXPERIMENTS = {
    'cifar-10': CIFAR10Experiment}


def init_logger(job_type, flags):
    dir_path = os.getcwd()
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    dir_str = '{}-{}-{}'.format(job_type, time_str, run_id)
    log_dir = os.path.join(dir_path, job_type, dir_str)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir, flags=flags)
    return logger


def get_attack(**kwargs):
    resol = {'cifar-10': 32}

    options = {
        'cutout': Cutout(kwargs['n_holes'], kwargs['length']),
        'patch_gaussian': PatchGaussian(kwargs['patch_size'], kwargs['max_scale'], kwargs['sample_up_to']),
        'pgd_linf': PGDAttack(
            norm='linf',
            nb_its=kwargs['n_iters'],
            eps_max=kwargs['epsilon'],
            step_size=kwargs['step_size'],
            resol=resol[kwargs['dataset']],
            scale_each=kwargs['scale_each'],
        ),
        'pgd_l2': PGDAttack(
            norm='l2',
            nb_its=kwargs['n_iters'],
            eps_max=kwargs['epsilon'],
            step_size=kwargs['step_size'],
            resol=resol[kwargs['dataset']],
            scale_each=kwargs['scale_each']
        ),
        'pgd_l1': PGDAttack(
            norm='l1',
            nb_its=kwargs['n_iters'],
            eps_max=kwargs['epsilon'],
            step_size=kwargs['step_size'],
            resol=resol[kwargs['dataset']],
            scale_each=kwargs['scale_each'])
    }

    return options[kwargs['attack_name']]


class Engine(object):
    def __init__(self, **config):
        FLAGS = FlagHolder()
        FLAGS.initialize(**config)
        if FLAGS.step_size is None:
            print('entered')
            FLAGS.step_size = pyt_common.get_step_size(FLAGS.epsilon, FLAGS.n_iters, FLAGS.use_max_step)
            FLAGS._dict['step_size'] = FLAGS.step_size
        FLAGS.summary()

        logger = init_logger('train', FLAGS._dict)

        if FLAGS.checkpoint_dir is None:
            FLAGS.checkpoint_dir = logger.log_dir
        print('checkpoint at {}'.format(FLAGS.checkpoint_dir))

        config['logger'] = logger
        config['step_size'] = FLAGS.step_size  # must be set before creating attack.
        config['attack'] = get_attack(**config)

        model = pyt_common.get_model(FLAGS.dataset, FLAGS.resnet_size, 1000 // FLAGS.class_downsample_factor)
        self.trainer = EXPERIMENTS[config['dataset']](model=model, **config)

    def run(self):
        self.trainer.train()
        print("Training finished.")


@click.command()
# training options:
@click.option('--mode', default='advers')
# dataset options:
@click.option('--dataset', default='cifar-10')
@click.option('--dataset_path', default='.', help="Location of the training data")
# model options:
@click.option("--resnet_size", default=50)
@click.option("--class_downsample_factor", default=1, type=int)
# optimization options:
@click.option('--batch_size', default=32)
@click.option("--epochs", default=200)
@click.option("--base_lr", default=0.1, type=float)
@click.option("--momentum", default=0.9, type=float)
@click.option("--wd", default=1e-4, type=float)
# attack options:
@click.option('--attack_name', default='cutout')
# pgd:
@click.option('--n_iters', default=10, type=int)
@click.option('--epsilon', default=16.0, type=float)
@click.option('--step_size', default=None, type=float)
@click.option('--scale_each', default=True, type=bool)
@click.option('--rand_target', default=True, type=bool)
@click.option('--scale_eps', default=True, type=bool)
# cutout:
@click.option("--n_holes", default=1, type=int)
@click.option("--length", default=16, type=int)
# patch gaussian:
@click.option("--patch_size", default=16, type=int)
@click.option("--max_scale", default=1, type=int)
@click.option("--sample_up_to", default=False, type=bool)
# output options:
@click.option("--checkpoint_dir", default=None, help="Location to write the final ckpt to")
# don't know:
@click.option("--step_size", default=None, type=float)  # alpha in pgd. depends on n_iters and epsilon.
@click.option("--use_max_step", is_flag=True, default=False)
def main(**flags):
    engine = Engine(**flags)
    engine.run()


if __name__ == '__main__':
    main()
