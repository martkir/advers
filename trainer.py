import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms


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


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def correct(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float()


class BaseTrainer(object):
    # Notes:
    # The attack needs to be initialized after the cuda device is set
    def __init__(self, batch_size=32, base_lr=0.0125, momentum=0.9, wd=1e-4, epochs=90, warmup_epochs=5, stride=10,
                 label_smoothing=-1.0, rand_target=False, run_val=True, model=None, checkpoint_dir=None,
                 dataset_path='/mnt/imagenet-test/', attack=None, attack_backward_steps=0, attack_loss='avg',
                 scale_eps=False, rand_init=True, logger=None):

        # Training options:
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.momentum = momentum
        self.wd = wd
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.stride = stride
        self.label_smoothing = label_smoothing
        self.rand_target = rand_target

        # Validation options:
        self.run_val = run_val

        # Model/checkpoints options
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.dataset_path = dataset_path

        # Attack options
        self.attack = attack
        self.attack_backward_steps = attack_backward_steps
        self.attack_loss = attack_loss
        self.scale_eps = scale_eps
        self.rand_init = rand_init

        # Logging options
        self.logger = logger

        # Dataset instance args. Set by subclass e.g. CIFAR10Trainer
        self.train_loader = None
        self.val_loader = None
        self.val_dataset = None

        assert self.attack_loss in ['avg', 'adv_only', 'logsumexp', 'max']
        assert self.model is not None

        # Set up checkpointing
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cuda = True
        self.verbose = 1

        if self.verbose:
            print(self.model)

        if self.cuda:
            self.model.cuda()

        if self.attack:
            self.attack = self.attack()
            self.attack_backward_steps = self.attack.nb_backward_steps

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._init_loaders()
        self._init_optimizer()

    def _init_loaders(self):
        raise NotImplementedError

    def _init_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=self.momentum,
                                   weight_decay=self.wd)

    def _checkpoint(self, epoch):
        if self.checkpoint_dir:
            out_fname = '{:02d}.pth'.format(epoch)
            out_fname = os.path.join(self.checkpoint_dir, out_fname)
            state = {'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
            torch.save(state, out_fname)

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
        raise NotImplementedError

    def _train_epoch(self, epoch):
        """
        note: train_loader, val_dataset are set by child class.
        """
        self.model.train()

        train_std_loss = Metric('train_std_loss')
        train_std_acc = Metric('train_std_acc')
        train_adv_loss = Metric('train_adv_loss')
        train_adv_acc = Metric('train_adv_acc')

        if self.attack:
            self.attack.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            print('batch_idx: ', batch_idx)
            if self.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            self._adjust_learning_rate(epoch)
            loss = torch.zeros([], dtype=torch.float32, device='cuda')
            if (not self.attack) or self.attack_loss == 'avg':
                output = self.model(data)
                loss += self._compute_loss(output, target)
                train_std_loss.update(loss)
                train_std_acc.update(accuracy(output, target))
            else:
                with torch.no_grad():
                    self.model.eval()
                    output = self.model(data)
                    train_std_loss_val = self._compute_loss(output, target)
                    train_std_loss.update(train_std_loss_val)
                    train_std_acc.update(accuracy(output, target))
                    self.model.train()

            if self.attack:
                if self.rand_target:
                    attack_target = torch.randint(0, len(self.val_dataset.classes) - 1, target.size(),
                                                  dtype=target.dtype, device='cuda')
                    attack_target = torch.remainder(target + attack_target + 1, len(self.val_dataset.classes))

                if self.rand_target:
                    data_adv = self.attack(self.model, data, attack_target, avoid_target=False,
                                           scale_eps=self.scale_eps)
                else:
                    data_adv = self.attack(self.model, data, target,
                                           avoid_target=True, scale_eps=self.scale_eps)
                output_adv = self.model(data_adv)
                adv_loss = self._compute_loss(output_adv, target)

                train_adv_loss.update(adv_loss)
                train_adv_acc.update(accuracy(output_adv, target))
                loss += adv_loss
                if self.attack_loss == 'avg':
                    loss /= 2.

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        log_dict = {'train_std_loss': train_std_loss.avg.item(),
                    'train_std_acc': train_std_acc.avg.item(),
                    'train_adv_loss': train_adv_loss.avg.item(),
                    'train_adv_acc': train_adv_acc.avg.item()}
        print(log_dict)
        self.logger.log(log_dict, epoch)

    def _val_epoch(self, epoch):
        self.model.eval()

        val_std_loss = Metric('val_std_loss')
        val_std_acc = Metric('val_std_acc')

        val_adv_acc = Metric('val_adv_acc')
        val_adv_loss = Metric('val_adv_loss')
        val_max_adv_acc = Metric('val_max_adv_acc')
        val_max_adv_loss = Metric('val_max_adv_loss')

        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(data)
                val_std_loss.update(F.cross_entropy(output, target))
                val_std_acc.update(accuracy(output, target))
            if self.attack:
                rand_target = torch.randint(
                    0, len(self.val_dataset.classes) - 1, target.size(),
                    dtype=target.dtype, device='cuda')
                rand_target = torch.remainder(target + rand_target + 1, len(self.val_dataset.classes))
                data_adv = self.attack(self.model, data, rand_target,
                                       avoid_target=False, scale_eps=self.scale_eps)
                data_max_adv = self.attack(self.model, data, rand_target, avoid_target=False, scale_eps=False)
                with torch.no_grad():
                    output_adv = self.model(data_adv)
                    val_adv_loss.update(F.cross_entropy(output_adv, target))
                    val_adv_acc.update(accuracy(output_adv, target))

                    output_max_adv = self.model(data_max_adv)
                    val_max_adv_loss.update(F.cross_entropy(output_max_adv, target))
                    val_max_adv_acc.update(accuracy(output_max_adv, target))
            self.model.eval()

        log_dict = {'val_std_loss': val_std_loss.avg.item(),
                    'val_std_acc': val_std_acc.avg.item(),
                    'val_adv_loss': val_adv_loss.avg.item(),
                    'val_adv_acc': val_adv_acc.avg.item(),
                    'val_max_adv_loss': val_max_adv_loss.avg.item(),
                    'val_max_adv_acc': val_max_adv_acc.avg.item()}
        self.logger.log(log_dict, epoch)

        if self.verbose:
            print(log_dict)

        self.optimizer.zero_grad()

    def train(self):
        print('Beginning training with {} epochs'.format(self.epochs))

        for epoch in range(self.epochs):
            begin = time.time()
            self._train_epoch(epoch)
            if self.run_val:
                self._val_epoch(epoch)
            self._checkpoint(epoch)
            end = time.time()
            if self.verbose:
                print('Epoch {} took {:.2f} seconds'.format(epoch, end - begin))

        self.logger.log_ckpt(self.model, self.optimizer)
        self.logger.end(summarize_vals=True)


class CIFAR10Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        if 'epochs' not in kwargs:
            kwargs['epochs'] = 200
        if 'base_lr' not in kwargs:
            kwargs['base_lr'] = 0.1
        super().__init__(**kwargs)

    def _adjust_learning_rate(self, epoch):
        if epoch < 100:
            lr = 0.1
        elif epoch < 150:
            lr = 0.01
        else:
            lr = 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _init_loaders(self):

        self.train_dataset = datasets.CIFAR10(
                root=self.dataset_path, download=True, train=True,
                transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        self.normalize]))

        self.val_dataset = datasets.CIFAR10(
                root=self.dataset_path,
                train=False,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        self.normalize]))

        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                shuffle=False, pin_memory=True)