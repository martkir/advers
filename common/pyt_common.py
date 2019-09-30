import math
from torchvision import models
from models import cifar10_resnet


def get_imagenet_model(resnet_size, nb_classes):
    size_to_model = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152
    }
    return size_to_model[resnet_size](num_classes=nb_classes)


def get_cifar10_model(resnet_size):
    size_to_model = {
        20: cifar10_resnet.resnet20,
        32: cifar10_resnet.resnet32,
        44: cifar10_resnet.resnet44,
        56: cifar10_resnet.resnet56,
        110: cifar10_resnet.resnet110,
    }
    model = size_to_model[resnet_size]()
    return model


def get_model(dataset, resnet_size, nb_classes):
    if dataset in ['imagenet', 'imagenet-c']:
        return get_imagenet_model(resnet_size, nb_classes)
    elif dataset in ['cifar-10', 'cifar-10-c']:
        return get_cifar10_model(resnet_size)


def get_step_size(epsilon, n_iters, use_max=False):
    if use_max:
        return epsilon
    else:
        return epsilon / math.sqrt(n_iters)
