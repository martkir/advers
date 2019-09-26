import json
import os
import torch


class Logger(object):
    def __init__(self, job_type, run_id, log_dir=None, flags=None):
        self.job_type = job_type
        self.log_dir = log_dir
        self.flags = flags

        if job_type == 'eval':
            self.img_dir = os.path.join(self.log_dir, 'images')
            if self.img_dir is not None:
                os.mkdir(self.img_dir)
        self.vals = {}
        self.summary = {}

        self.log_summary(flags)
        if self.job_type == 'train':
            self.log_summary({'run_id': run_id})

    def log(self, log_dict, step):
        for k, v in log_dict.items():
            if k not in self.vals:
                self.vals[k] = []
            self.vals[k].append((v, step))

    def log_image(self, image, filestr):
        img_file = os.path.join(self.img_dir, filestr)
        image.save(img_file)

    def log_summary(self, log_dict):
        for k, v in log_dict.items():
            self.summary[k] = v

    def log_ckpt(self, model, optim):
        state = {'model': model.state_dict(),
                 'optimizer': optim.state_dict()}
        # automatically uploaded to wandb after the run
        torch.save(state, os.path.join(self.log_dir, 'ckpt.pth'))

    def write(self):
        all_file = os.path.join(self.log_dir, 'all.log')
        with open(all_file, 'w') as f:
            json.dump(self.vals, f)

    def write_summary(self):
        summary_file = os.path.join(self.log_dir, 'summary.log')
        with open(summary_file, 'w') as f:
            json.dump(self.summary, f)

    def end(self, summarize_vals=False):
        if summarize_vals:
            summary_dict = {}
            for k, v in self.vals.items():
                summary_dict[k] = v[-1][0]
            self.log_summary(summary_dict)
        self.write_summary()