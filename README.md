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

Each time an epoch is completed any output generated during that epoch is saved. The output of a training experiment is stored in `train/train-[id]/checkpoints` (where `[id]` is a unique sequence of characters generated). The content of this folder is as follows.

* `checkpoints` the folder where the model paramters and optimizer are saved every epoch.
* `config.csv` stores all the command line arguments that were used to run the training experiment.
* `all.csv` contains all statistics that were generated during a training experiment. This includes training (validation) loss, accuracy, both clean and adversarial.
* `images` contains plots of certain statistics e.g. train and valid accuracy vs. epochs.

