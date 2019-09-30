import click
from trainer import CIFAR10Experiment
from common import pyt_common
from common.flags_holder import FlagHolder

BUILTIN_TRAINERS = {
    'cifar-10': CIFAR10Experiment}


class Engine(object):
    def __init__(self, **config):
        FLAGS = FlagHolder()
        FLAGS.initialize(**config)
        if FLAGS.step_size is None:
            FLAGS.step_size = pyt_common.get_step_size(FLAGS.epsilon, FLAGS.n_iters, FLAGS.use_max_step)
            FLAGS._dict['step_size'] = FLAGS.step_size
        FLAGS.summary()

        logger = pyt_common.init_logger('train', FLAGS._dict)
        if FLAGS.checkpoint_dir is None:
            FLAGS.checkpoint_dir = logger.log_dir
        print('checkpoint at {}'.format(FLAGS.checkpoint_dir))

        model = pyt_common.get_model(FLAGS.dataset, FLAGS.resnet_size, 1000 // FLAGS.class_downsample_factor)
        attack = None
        if config['adv_train']:
            attack = pyt_common.get_attack(**config)

        self.trainer = BUILTIN_TRAINERS[config['dataset']](
            # model/checkpoint options
            model=model, checkpoint_dir=FLAGS.checkpoint_dir, dataset_path=FLAGS.dataset_path,
            # attack options
            attack=attack, scale_eps=FLAGS.scale_eps, attack_loss=FLAGS.attack_loss,
            # training options
            batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, stride=FLAGS.class_downsample_factor,
            label_smoothing=FLAGS.label_smoothing, rand_target=FLAGS.rand_target,
            # data options:
            dataset=FLAGS.dataset,
            pre_augment=FLAGS.pre_augment,
            preprocess_options=FLAGS.preprocess_options,
            # logging options
            logger=logger)

    def run(self):
        self.trainer.train()
        print("Training finished.")


@click.command()
# Dataset options ('cifar-10', 'imagenet'):
@click.option('--dataset', default='cifar-10')
@click.option("--dataset_path", default='.', help="Location of the training data")
@click.option("--pre_augment", default=True, type=bool, help="Whether to train in pre-augment mode.")
@click.option("--preprocess_options", default='standard normalize', type=str,
              help="The augmentations to apply for data preprocessing")
# Model options
@click.option("--resnet_size", default=50)
@click.option("--class_downsample_factor", default=1, type=int)
# Training options
@click.option("--batch_size", default=32)
@click.option("--epochs", default=90)
@click.option("--label_smoothing", default=0.0)
@click.option("--checkpoint_dir", default=None, help="Location to write the final ckpt to")
@click.option("--use_fp16/--no_fp16", is_flag=True, default=False)
# Adversarial training options
@click.option("--adv_train", default=False)
@click.option("--attack_loss", default='adv_only')  # 'avg', 'adv_only', or 'logsumexp'
@click.option("--rand_target/--no_rand_target", is_flag=True, default=True)
# Attack options:
# ['pgd_linf', 'pgd_l2', 'fw_l1', 'jpeg_linf', 'jpeg_l2', 'jpeg_l1', 'elastic', 'fog', 'gabor', 'snow', 'cutout']
@click.option("--attack_name", default=None, type=str)
@click.option("--epsilon", default=16.0, type=float)
@click.option("--step_size", default=None, type=float)
@click.option("--use_max_step", is_flag=True, default=False)
@click.option("--n_iters", default=10, type=int)
@click.option("--scale_each/--no_scale_each", is_flag=True, default=True)
@click.option("--scale_eps/--no_scale_eps", is_flag=True, default=True)
# Cutout attack specific options:
@click.option("--n_holes", default=1, type=int)
@click.option("--length", default=16, type=int)
# PatchGaussian specific options:
@click.option("--patch_size", default=16, type=int)
@click.option("--max_scale", default=1, type=int)
@click.option("--sample_up_to", default=False, type=bool)
def main(**flags):
    engine = Engine(**flags)
    engine.run()


if __name__ == '__main__':
    main()
