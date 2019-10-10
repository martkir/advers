import os
import torch
import pandas as pd
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, log_dir=None, flags=None):
        self.log_dir = log_dir  # e.g. train/train-20190928_173723-3b129d10
        self.flags = flags

        # directory where images are saved:
        self.img_dir = os.path.join(self.log_dir, 'images')
        if self.img_dir is not None:
            os.mkdir(self.img_dir)

        # vals saved epoch results.
        self.vals = {}

        # config_dict saves experiment arguments e.g. hyperparameters and training options.
        self.config_dict = {}
        self._write_config(flags)

    def add(self, log_dict):  # done.
        """ Adds data to log_dict but does not write to folder.
        """
        for k, v in log_dict.items():
            if k not in self.vals:
                self.vals[k] = []
            self.vals[k].append(v)

    def _write_config(self, flags, extension='csv'):  # done.
        for k, v in flags.items():
            self.config_dict[k] = v

        df = pd.DataFrame(self.config_dict, index=[0])
        if extension == 'json':
            config_file = os.path.join(self.log_dir, 'config.json')
            df.to_json(config_file, index=False)
        else:
            config_file = os.path.join(self.log_dir, 'config.csv')
            df.to_csv(config_file, index=False, header=True, sep=',')

    def write_vals(self, extension='csv'):  # done.
        df = pd.DataFrame.from_dict(self.vals)
        if extension == 'json':
            all_file = os.path.join(self.log_dir, 'all.json')
            df.to_json(all_file, index=False)
        else:
            all_file = os.path.join(self.log_dir, 'all.csv')
            df.to_csv(all_file, index=False, header=True, sep=',')

    def write_plot(self, plot_name, keys=None, description=None):
        n_epochs = len(self.vals[list(self.vals.keys())[0]])
        epochs = [i for i in range(n_epochs)]
        fig, ax = plt.subplots(1, 1)

        if keys is None:
            for k, vals in self.vals.items():
                ax.plot(epochs, vals, linewidth=1, label=k)
        else:
            for k in keys:
                ax.plot(epochs, self.vals[k], linewidth=1, label=k)

        if description:
            ax.set_xlabel(description)

        ax.legend(loc=2)
        fig.set_figheight(5.5)
        fig.set_figwidth(11)
        plt.savefig(os.path.join(self.img_dir, '{}.png'.format(plot_name)))

    def save_ckpt(self, model, optim):
        """ This is called after the last epoch has finished. A different method is used to save checkpoints after
        each epoch.
        """
        state = {'model': model.state_dict(),
                 'optimizer': optim.state_dict()}
        torch.save(state, os.path.join(self.log_dir, 'ckpt.pth'))