### Normal & Adversarial Training

The module `train.py` can be used for both normal and adversarial training. For example:

`python train.py --mode normal --dataset cifar-10 --dataset_path data/cifar-10 --normal_aug standard-normalize --advers_aug advers-standard-normalize --resnet_size 20 --base_lr 0.1 --wd 1e-4 --momentum 0.9 --batch_size 128 --epochs 100 -attack_name cutout --n_holes 1 --length 16`

will train a ResNet20 model using SGD with weight decay and momentum on CIFAR-10 for 100 epochs.

Training options:

* `--mode` determines how a model is trained. Options: `normal` and `advers`. If `normal` the model is trained on "clean" data; if `advers` on adversarial data.

Data options:

* `dataset` specifies the dataset to train a model on. Options: `cifar-10`.
* `dataset_path` the location to save the data to (if not already there).
* `normal_aug` the augmentation(s) to apply to the clean data. Default: `standard-normalize`. This performs random crop, horizontal flip, and normalization to the clean data.
* `advers_aug` the augmentation(s) for creating an adversarial attack. Default: `advers-standard-normalize`. The default is to augment the data by first performing an adversarial attack on it; after which it is randomly cropped, horizontally flipped and normalized.

Model options:
* `resnet_size` determines which ResNet model is used.

Optimization options:

The optimizer that is used is SGD with weight decay, and momentum. Flags `base_lr`, `wd` and `momentum` are used to specify the corresponding hyperparameters.

* `batch_size` the batch size to use.
* `epochs` the number of epochs to train a model for.

Output:

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

