import copy
import functools
import os
import time
from tqdm import tqdm

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils import logger
from utils.fp16_util import MixedPrecisionTrainer
from models.nn import update_ema
from models.resample import LossAwareSampler, UniformSampler


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            val_data,
            batch_size,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            val_interval,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_decay_steps=0,
            lr_decay_factor = 1,
            main_data_indentifier_input: str = "input",
            main_data_indentifier_trans: str = "trans",
            iterations: int = 70e3,
            num_samples=None,
            num_input_channels=None,
            image_size=None,
            clip_denoised=None,
            use_ddim=None,
            device = None,
            args = None
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_factor = lr_decay_factor
        self.main_data_indentifier_input = main_data_indentifier_input
        self.main_data_indentifier_trans = main_data_indentifier_trans
        self.iterations = iterations
        self.num_samples = num_samples
        self.num_input_channels = num_input_channels
        self.image_size = image_size
        self.clip_denoised = clip_denoised
        self.use_ddim = use_ddim
        self.t = None

        log_dir = os.path.join('../../logs_loss/', 'train')
        self.writer = SummaryWriter(log_dir=log_dir)
        '''timing'''
        self.args = args
        self.step = 0
        self.time_iter_start = 0
        self.forward_backward_time = 0
        self.device = device
        self.x0_pred = None
        self.recursive_flag = 0
        self.resume_step = 0
        self.sync_cuda = th.cuda.is_available()
        self.current_loss = 0

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.ema_params = [
            copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate))
        ]

    def run_loop(self):
        while (
                not self.lr_decay_steps
                or self.step + self.resume_step < self.iterations
        ):
            data_dict = next(self.data)
            self.run_step(data_dict, self.step)
            if self.step % self.save_interval == 0:
                self.save()
            if self.step % self.log_interval == 0:
                self.time_iter_end = time.time()
                if self.time_iter_start == 0:
                    self.time_iter = 0
                else:
                    self.time_iter = self.time_iter_end - self.time_iter_start
                self.log_step()
                logger.dumpkvs()
                self.time_iter_start = time.time()
            if self.step % self.val_interval == 0:  # Validation interval
                self.run_validation()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def run_validation(self):
        val_losses = []
  
        print(type(self.val_data))
        for i, val_data_dict in tqdm(enumerate(self.val_data), total = len(self.val_data)):
            
            val_loss = self.run_step(val_data_dict, self.step, phase="val")
           
            val_losses.append(val_loss.detach().cpu().numpy())
        avg_val_loss = np.mean(val_losses)
        self.writer.add_scalar('Val_Loss', avg_val_loss, self.step)
        print(f"Validation Loss at step {self.step}: {avg_val_loss}")

    def run_step(self, data_dict, iteration, phase="train"):
        self.forward_backward(data_dict, iteration, phase=phase)
        if phase == "train":
            took_step = self.mp_trainer.optimize(self.opt)
            if took_step:
                self._update_ema()
            self.lr_decay()
        return self.current_loss


    def forward_backward(self, data_dict, iteration, phase: str = "train"):

        if self.recursive_flag == 0:
            self.batch_image_input = data_dict.pop(self.main_data_indentifier_input)

            # self.batch_image_seg = data_dict.pop('seg')
            # self.brain_mask = th.ones(self.batch_image_seg.shape) * (data_dict.pop('brainmask') > 0)

            self.batch_image_input = self.batch_image_input.to(self.device)  # t1
            self.batch_masks = data_dict.pop("input_mask_tolerated")
            self.batch_masks = self.batch_masks.to(self.device)

            self.noncon_arota = data_dict.pop("input_contrast").to(self.device)
            self.noncon_arota_hist = data_dict.pop("trans_hist").to(self.device)
            self.con_arota_hist = data_dict.pop("input_hist").to(self.device)

            # self.batch_image_seg = self.batch_image_seg.to(self.device)  # seg
            # self.brain_mask = self.brain_mask.to(self.device)
            self.model_conditionals = data_dict

        assert phase in ["train", "val"]

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        self.mp_trainer.zero_grad()


        self.t, self.weights = self.schedule_sampler.sample(self.batch_image_input.shape[0], self.device)

        x0_t = None
        labels = None
        self.model_conditionals = labels
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            model=self.model,
            input_img=self.noncon_arota,
            trans_img=self.batch_image_input,
            aneurysm_mask = self.batch_masks,
            noncon_arota_hist=self.noncon_arota_hist,
            con_arota_hist=self.con_arota_hist,
            have_noncon_arota_hist = self.args.noncontrast_hist,
            have_con_arota_hist = self.args.contrast_hist,
            model_name = self.args.model_name,
            t=self.t,
            iteration=iteration,
            x_start_t=x0_t,
            model_kwargs=self.model_conditionals,
            noise = None
        )

        losses = compute_losses()


        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                self.t, losses["loss"].detach()
            )

        loss = losses["loss"].mean()
        log_loss_dict(
            self.diffusion, self.t, {phase + '_' + k: v * self.weights for k, v in losses.items()}
        )
        if phase == "train":
            self.mp_trainer.backward(loss)

            '''plot training loss'''
            self.writer.add_scalar('Loss', loss.detach().cpu().numpy(), self.step)
        self.current_loss = loss


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def lr_decay(self):
        if self.lr_decay_steps == 0 or self.step % self.lr_decay_steps != 0 or self.step == 0:
            return
        print('lr decay.....')
        n_decays = self.step // self.lr_decay_steps
        lr = self.lr * self.lr_decay_factor ** n_decays
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1))
        logger.logkv("time 100iter", self.time_iter)

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model ...")
            filename = f"model{(self.step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None



def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
