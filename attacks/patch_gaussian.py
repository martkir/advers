import torch
import torch.nn as nn
import numpy as np


class PatchGaussian(nn.Module):
    def __init__(self, patch_size, max_scale, sample_up_to=False):
        """
        Args:
            patch_size: The size of a patch. The patch is square.
            max_scale: The maximum Gaussian noise to apply.
            sample_up_to: If True makes the uniformly at random makes mask size be between 1 and patch_size.
        """
        super(PatchGaussian, self).__init__()
        self.patch_size = patch_size
        self.max_scale = max_scale
        self.sample_up_to = sample_up_to
        self.attack_name = 'patch_gaussian'

    def forward(self, img):
        """
        Args: img (Tensor): Image. Shape (n_channels, height, width). Pixel values are assumed to be between [0, 255].
        """

        if self.sample_up_to:
            patch_size = np.random.randint(1, self.patch_size + 1)
            # otherwise, patch_size is fixed.

        # make image (which is [0, 255]) be [0, 1]
        img = img / 255.0

        # uniformly sample scale from 0 to given scale
        scale = self.max_scale * np.random.uniform(0, 1)

        # apply patch gaussian noise to image:
        mask = self._get_patch_mask(img, inverse=True)
        noise = scale * torch.randn(img.shape)
        noise = noise * mask
        img_plus_noise = torch.clamp(img + noise, min=0, max=1) * 255 # scale back to [0, 255].

        return img_plus_noise

    def _get_patch_mask(self, img, inverse=False):

        # randomly sample location in image:
        n_channels, img_height, img_width = img.shape
        x = np.random.randint(0, img_width + 1)
        y = np.random.randint(0, img_height + 1)

        # compute where the patch will start and end.
        start_x = int(np.max([x - np.floor(self.patch_size / 2), 0]))
        end_x = int(np.min([x + np.ceil(self.patch_size / 2), img_width]))

        start_y = int(np.max([y - np.floor(self.patch_size / 2), 0]))
        end_y = int(np.min([y + np.ceil(self.patch_size / 2), img_height]))

        # create mask
        if inverse:
            mask = torch.zeros(n_channels, img_height, img_width)
            mask[:, start_y: end_y, start_x: end_x] = 1
        else:
            mask = torch.ones(n_channels, img_height, img_width)
            mask[:, start_y: end_y, start_x: end_x] = 0

        return mask

    def set_epoch(self, epoch):
        """ Must be implemented in order to use attack with <trainer.py>.
        """
        self.epoch = epoch
        self._update_params(epoch)

    def _update_params(self, epoch):
        pass