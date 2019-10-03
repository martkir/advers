# advers

### Normal & Adversarial Training

The module `train.py` can be used for both normal and adversarial training. For example:

`python train.py --mode normal --normal_aug standard-normalize --advers_aug advers-standard-normalize --dataset cifar-10 --dataset_path data/cifar-10 --pre_augment True --preprocess_options advers-standard-normalize --resnet_size 20 --base_lr 0.1 --wd 1e-4 --momentum 0.9`

will train a ResNet20 model using SGD with weight decay and momentum on CIFAR-10 for 100 epochs.

Explanation of arguments:
* `--normal_aug` is used to specify the augmentation to apply to the clean data. The default setting is to apply random cropping,  horizontal flipping, and normalization.
* `--advers_aug` is the augmentations applied to the data in order to create an adversarial example. The default is `advers-standard-normalize`. An adversarial example is constructed by first applying an attack (specified by `--attack`), then the standard augmentations described above.
* If `--mode` is `normal` the model parameters are learened by using the clean data. The adversarial attack that is specified will merely be used to evaluate a model's robustness against said attack. If `--mode` is `advers` the model will be trained using on the adversarial examples using the attack specified by `--attack`.

Output of training:

The output of a training experiment is stored in `train/train-[id]/checkpoints`, where `[id]` is a unique sequence of characters generated each time a training experiment is ran. This folder stores the following three items:
* `checkpoints`. This is the folder where the model parameters and optimizer are saved every epoch.
* `images`

<!-- todo: explain that the trainer in normal mode still calculates advers accuracy. the only difference is not it doesn't train on
adversarial batches - just the clean ones. todo: preprocess_options flag should be split up into two: advers_options: , normal options: this way you can still specify the order of augmentations.
-->

<!-- `python eval.py --dataset imagenet --class_downsample_factor 10 --attack pgd_linf --epsilon 16.0 --n_iters 100 --step_size 1 --ckpt_path [CKPT_PATH] --dataset_path [DATASET_PATH]`)

will evaluate a ResNet-50 model checkpoint located at `CKPT_PATH` against the L<sub>&infin;</sub> attack with &epsilon;=16, 100 iterations, and step size 1 on the ImageNet-100 validation set, located at `DATASET_PATH`.  The choices of attack we provide are: `pgd_linf, pgd_l2, fw_l1, jpeg_linf, jpeg_l2, jpeg_l1, elastic, fog, gabor, snow`.

If the flag `--use_wandb` is set, results will be logged to WandB.  Otherwise, if the flag `--no_wandb` is set, results will be logged to the folder `./eval/eval-[YYYYmmDD_HHMMSS]-[RUN_ID]`, which will contain:

* file `summary.log`, a JSON file containing a single dict with configuration parameters and results of the run
* folder `images` containing (1) `adv_[X].png`, `orig_[X].png`, attacked and original version of the first image in each class X and (2) `init_adv_[X].png`, `init_orig_[X].png`, attacked and original versions of all images in the first evaluation batch.) -->
