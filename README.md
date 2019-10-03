# advers

### Adversarial and normal training

The module  `train.py` can be used to train a model; both adversarially and normally. Example:

`python train.py --dataset cifar-10 --dataset_path data/cifar-10 --pre_augment True --adv_train True --preprocess_options advers-standard-normalize --resnet_size 20 --base_lr 0.1 --wd 1e-4 --momentum 0.9`

will adversarially train a ResNet20 model using SGD with weight decay and momentum on CIFAR-10 for 100 epochs.

Remarks:

* `preprocess_options` is used to specify the order in which the data augmentations are applied. In the above example, during training, the images in a batch are transformed by first adversarially attacking them, then applying standard data augmentation (i.e. random crop, and horizontal flip), after finally normalizing them.

<!-- todo: explain that the trainer in normal mode still calculates advers accuracy. the only difference is not it doesn't train on
adversarial batches - just the clean ones. todo: preprocess_options flag should be split up into two: advers_options: , normal options: this way you can still specify the order of augmentations.
-->

<!-- `python eval.py --dataset imagenet --class_downsample_factor 10 --attack pgd_linf --epsilon 16.0 --n_iters 100 --step_size 1 --ckpt_path [CKPT_PATH] --dataset_path [DATASET_PATH]`)

will evaluate a ResNet-50 model checkpoint located at `CKPT_PATH` against the L<sub>&infin;</sub> attack with &epsilon;=16, 100 iterations, and step size 1 on the ImageNet-100 validation set, located at `DATASET_PATH`.  The choices of attack we provide are: `pgd_linf, pgd_l2, fw_l1, jpeg_linf, jpeg_l2, jpeg_l1, elastic, fog, gabor, snow`.

If the flag `--use_wandb` is set, results will be logged to WandB.  Otherwise, if the flag `--no_wandb` is set, results will be logged to the folder `./eval/eval-[YYYYmmDD_HHMMSS]-[RUN_ID]`, which will contain:

* file `summary.log`, a JSON file containing a single dict with configuration parameters and results of the run
* folder `images` containing (1) `adv_[X].png`, `orig_[X].png`, attacked and original version of the first image in each class X and (2) `init_adv_[X].png`, `init_orig_[X].png`, attacked and original versions of all images in the first evaluation batch.) -->
