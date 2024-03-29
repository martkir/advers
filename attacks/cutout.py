import torch
import torch.nn as nn
import numpy as np


class Cutout(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.

    Note: This code was written to be applied on a
    """
    def __init__(self, n_holes, length):
        super(Cutout, self).__init__()

        self.n_holes = n_holes
        self.length = length
        self.attack_name = 'cutout'

    def forward(self, img):
        """
        Args:
            img (Tensor): Image of shape (n_channels, height, width).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.  # binary matrix shape (height, width).

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

    def set_epoch(self, epoch):
        """ Must be implemented in order to use attack with <trainer.py>.
        """
        self.epoch = epoch
        self._update_params(epoch)

    def _update_params(self, epoch):
        pass
