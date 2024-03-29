import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


class DatasetAdversCIFAR10(datasets.CIFAR10):
    def __init__(self, transform_adv=None, **kwargs):
        super(DatasetAdversCIFAR10, self).__init__(**kwargs)
        self.transform_adv = transform_adv

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img_adv = img

        if self.transform is not None:
            img = self.transform(img)

        if self.transform_adv is not None:
            img_adv = self.transform_adv(img_adv)

        if self.target_transform is not None:
            target = self.target_transform(target)

        output = (img, img_adv, target)
        return output


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def correct(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float()


class BaseTrainer(object):
    def __init__(self, **kwargs):
        self.train_loader = kwargs['train_loader']
        self.train_dataset = kwargs['train_dataset']
        self.val_loader = kwargs['val_loader']
        self.val_dataset = kwargs['val_dataset']

        self.mode = kwargs['mode']
        self.cuda = kwargs['cuda']
        self.logger = kwargs['logger']
        self.attack = kwargs['attack']

        # pgd:
        self.rand_target = kwargs['rand_target']
        self.scale_eps = kwargs['scale_eps']

        # idk:
        self.label_smoothing = 0

    def _compute_loss(self, output, target):
        if self.label_smoothing > 0:
            n_class = len(self.val_dataset.classes)
            one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot.clamp(self.label_smoothing / (n_class - 1), 1 - self.label_smoothing)
            log_prob = F.log_softmax(output, dim=1)
            loss = -(one_hot * log_prob).sum(dim=1)
            return loss.mean()
        else:
            return F.cross_entropy(output, target)

    def _adjust_learning_rate(self, epoch):
        pass

    def _get_batch(self, *args, **kwargs):
        raise NotImplementedError

    def train_epoch(self, epoch, model, optimizer):
        model.train()
        train_adv_loss = Metric('train_adv_loss')
        train_adv_acc = Metric('train_adv_acc')
        train_std_loss = Metric('train_std_loss')
        train_std_acc = Metric('train_std_acc')

        if self.attack:
            self.attack.set_epoch(epoch)

        with tqdm(total=len(self.train_loader)) as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                data, data_adv, target = self._get_batch(model, batch)
                self._adjust_learning_rate(epoch)
                loss = torch.zeros([], dtype=torch.float32, device='cuda')

                if self.mode == 'normal':
                    # compute adversarial accuracy & loss:
                    model.eval()
                    output_adv = model(data_adv)
                    adv_loss = self._compute_loss(output_adv, target)
                    train_adv_loss.update(adv_loss)
                    train_adv_acc.update(accuracy(output_adv, target))

                    # compute clean accuracy & loss (train mode):
                    model.train()
                    output = model(data)
                    std_loss = self._compute_loss(output, target)
                    train_std_loss.update(std_loss)
                    train_std_acc.update(accuracy(output, target))
                    loss += std_loss  # update using clean loss.

                else:

                    # compute clean accuracy & loss (eval mode):
                    model.eval()
                    output = model(data)
                    train_std_loss_val = self._compute_loss(output, target)
                    train_std_loss.update(train_std_loss_val)
                    train_std_acc.update(accuracy(output, target))

                    # compute adversarial accuracy & loss (train mode):
                    model.train()
                    output_adv = model(data_adv)
                    adv_loss = self._compute_loss(output_adv, target)
                    train_adv_loss.update(adv_loss)
                    train_adv_acc.update(accuracy(output_adv, target))
                    loss += adv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_stats = {'train_std_loss': train_std_loss.avg.item(),
                               'train_std_acc': train_std_acc.avg.item(),
                               'train_adv_loss': train_adv_loss.avg.item(),
                               'train_adv_acc': train_adv_acc.avg.item()}

                description = 'epoch: {} '.format(epoch)
                description += ' '.join(['{}: {:.4f}'.format(k, v) for k, v in batch_stats.items()])

                pbar.update(1)
                pbar.set_description(description)

        log_dict = {'train_std_loss': train_std_loss.avg.item(),
                    'train_std_acc': train_std_acc.avg.item(),
                    'train_adv_loss': train_adv_loss.avg.item(),
                    'train_adv_acc': train_adv_acc.avg.item()}

        return model, optimizer, log_dict

    def val_epoch(self, epoch, model, optimizer, verbose=True):
        """
        This function iterates through the dataset. Each iteration processes a batch of data. The processing of a batch
        consists of:
        (1) Calculating the "clean" accuracy of our model on the batch.
        (2) Calculating the adversarial accuracy of our model on the batch.
        """

        model.eval()
        val_std_loss = Metric('val_std_loss')
        val_std_acc = Metric('val_std_acc')
        val_adv_acc = Metric('val_adv_acc')
        val_adv_loss = Metric('val_adv_loss')

        for batch_idx, batch in enumerate(self.val_loader):
            data, data_adv, target = self._get_batch(model, batch)

            with torch.no_grad():
                # Step 1: Calculating "clean" accuracy:
                output = model(data)
                val_std_loss.update(F.cross_entropy(output, target))
                val_std_acc.update(accuracy(output, target))

                # Step 2(a): Calculating adversarial accuracy:
                output_adv = model(data_adv)
                val_adv_loss.update(F.cross_entropy(output_adv, target))
                val_adv_acc.update(accuracy(output_adv, target))

            model.eval()

        log_dict = {'val_std_loss': val_std_loss.avg.item(),
                    'val_std_acc': val_std_acc.avg.item(),
                    'val_adv_loss': val_adv_loss.avg.item(),
                    'val_adv_acc': val_adv_acc.avg.item()}

        if verbose:
            print(log_dict)
        optimizer.zero_grad()

        return model, optimizer, log_dict


class TrainerCutout(BaseTrainer):
    def __init__(self, **kwargs):
        super(TrainerCutout, self).__init__(**kwargs)

    def _get_batch(self, model, batch):
        data, data_adv, target = batch
        if self.cuda:
            data, data_adv, target = data.cuda(non_blocking=True), data_adv.cuda(non_blocking=True), \
                                     target.cuda(non_blocking=True)

        return data, data_adv, target


class TrainerPGD(BaseTrainer):
    def __init__(self, **kwargs):
        super(TrainerPGD, self).__init__(**kwargs)

    def _get_batch(self, model, batch):
        data, target = batch
        if self.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        if self.rand_target:  # targeted attack. target chosen at random.
            attack_target = torch.randint(0, len(self.val_dataset.classes) - 1, target.size(), dtype=target.dtype,
                                          device='cuda')
            attack_target = torch.remainder(target + attack_target + 1, len(self.val_dataset.classes))
            data_adv = self.attack(model, data, attack_target, avoid_target=False, scale_eps=self.scale_eps)
        else:  # non_targeted attack.
            data_adv = self.attack(model, data, target, avoid_target=True, scale_eps=self.scale_eps)

        return data, data_adv, target


class BaseExperiment(object):

    def __init__(self, model, **kwargs):

        # training options:
        self.run_val = True
        self.mode = kwargs['mode']

        # optimization options
        self.batch_size = kwargs['batch_size']
        self.base_lr = kwargs['base_lr']
        self.momentum = kwargs['momentum']
        self.wd = kwargs['wd']
        self.epochs = kwargs['epochs']

        # Model/checkpoints options
        self.model = model
        self.checkpoint_dir = kwargs['checkpoint_dir']
        self.dataset_path = kwargs['dataset_path']

        # attack options:
        self.attack = kwargs['attack']
        self.attack_name = kwargs['attack_name']
        # pgd:
        self.n_iters = kwargs['n_iters']
        self.epsilon = kwargs['epsilon']
        self.step_size = kwargs['step_size']
        self.rand_target = kwargs['rand_target']
        self.scale_eps = kwargs['scale_eps']

        # logging options:
        self.logger = kwargs['logger']

        # dataset options:
        self.dataset = kwargs['dataset']

        # Dataset instance args. Set by subclass e.g. CIFAR10Trainer
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None
        self.optimizer = None

        assert self.model is not None

        # set up checkpointing:
        self.checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cuda = True
        self.verbose = 1

        if self.verbose:
            print(self.model)

        if self.cuda:
            self.model.cuda()

        assert self.attack, 'attack is {}'.format(self.attack)

        # note: trainer must be initialized after initializing the loaders.
        self._init_loaders()
        self._init_optimizer()
        self._init_trainer()

    def _init_loaders(self):
        raise NotImplementedError

    def _init_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=self.momentum,
                                   weight_decay=self.wd)

    def _init_trainer(self):
        raise NotImplementedError

    def _checkpoint(self, epoch):
        if self.checkpoint_dir:
            out_fname = '{:02d}.pth'.format(epoch)
            out_fname = os.path.join(self.checkpoint_dir, out_fname)
            state = {'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
            torch.save(state, out_fname)

    def train(self):
        print('Beginning training with {} epochs'.format(self.epochs))

        for epoch in range(self.epochs):
            begin = time.time()
            self.model, self.optimizer, train_log_dict = self.trainer.train_epoch(epoch, self.model, self.optimizer)
            self.logger.add(train_log_dict)

            if self.run_val:
                self.model, self.optimizer, val_log_dict = \
                    self.trainer.val_epoch(epoch, self.model, self.optimizer, self.verbose)
                self.logger.add(val_log_dict)

            self.logger.write_vals()
            self.logger.write_plot('std_train_vs_std_valid_acc', ['train_std_acc', 'val_std_acc'],
                                   description='Train vs. Valid Accuracy')
            self.logger.write_plot('train_adv_vs_val_adv_acc', ['train_adv_acc', 'val_adv_acc'],
                                   description='Adversarial Train vs. Valid Accuracy')
            self.logger.write_plot('val_adv_vs_val_std_acc', ['val_adv_acc', 'val_std_acc'],
                                   description='Adversarial Valid vs. Clean Accuracy')

            self._checkpoint(epoch)
            end = time.time()
            if self.verbose:
                print('Epoch {} took {:.2f} seconds'.format(epoch, end - begin))

        self.logger.save_ckpt(self.model, self.optimizer)


class CIFAR10Experiment(BaseExperiment):
    def __init__(self, **kwargs):
        super(CIFAR10Experiment, self).__init__(**kwargs)

    def _adjust_learning_rate(self, epoch):
        if epoch < 100:
            lr = 0.1
        elif epoch < 150:
            lr = 0.01
        else:
            lr = 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _init_loaders_pgd(self):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        clean_transform = transforms.Compose([transforms.ToTensor(), normalize])
        standard_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize])

        self.train_dataset = datasets.CIFAR10(root=self.dataset_path, download=True, train=True,
                                              transform=standard_transform)
        self.val_dataset = datasets.CIFAR10(root=self.dataset_path, train=False, transform=clean_transform)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True)

    def _init_loaders_cutout(self):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        clean_transform = transforms.Compose([transforms.ToTensor(), normalize])
        standard_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize])

        # advers attack:
        advers_transform = transforms.Compose([])
        advers_transform.transforms.append(transforms.ToTensor())
        advers_transform.transforms.append(self.attack)
        # standard augmentations:
        advers_transform.transforms.append(transforms.ToPILImage())
        advers_transform.transforms.append(transforms.RandomHorizontalFlip())
        advers_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        # normalize:
        advers_transform.transforms.append(transforms.ToTensor())
        advers_transform.transforms.append(normalize)

        self.train_dataset = DatasetAdversCIFAR10(root=self.dataset_path, download=True, train=True,
                                                  transform=standard_transform, transform_adv=advers_transform)
        self.val_dataset = DatasetAdversCIFAR10(root=self.dataset_path, train=False, transform=clean_transform,
                                                transform_adv=advers_transform)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True)

    def _init_loaders(self):
        if self.attack_name in ['cutout', 'patch_gaussian']:
            self._init_loaders_cutout()
        else:
            self._init_loaders_pgd()

    def _init_trainer(self):
        if self.attack_name in ['cutout', 'patch_gaussian']:
            self.trainer = TrainerCutout(**self.__dict__)
        else:
            self.trainer = TrainerPGD(**self.__dict__)
