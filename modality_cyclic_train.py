"""
Train a diffusion model on images.
"""
import time
import sys
import argparse
import os
import blobfile as bf
from pathlib import Path
import torch as th

from configs import get_config
from utils import logger
from datasets import loader
from models.resample import create_named_schedule_sampler
from utils.script_util import create_gaussian_diffusion, create_score_model
from utils.train_util import TrainLoop
import wandb

sys.path.append(str(Path.cwd()))


def main(args):
    use_gpus = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)
    time_load_start = time.time()
    config = get_config.file_from_dataset(args.dataset)
    
    wandb.init(project="oxaaaguideddif", config =config)

    wandb.config.update(vars(args))

    if args.experiment_name != 'None':
        experiment_name = args.experiment_name
    else:
        experiment_name = args.model_name + '_' + args.dataset + '_' + args.input + '_' + args.trans

    logger.configure(Path(experiment_name)/"score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    if args.model_name == 'unet':
        image_level_cond = False
    elif args.model_name == 'diffusion':
        image_level_cond = True
    else:
        raise Exception("Model name does exit")

    logger.configure(Path(experiment_name) / "score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])
    contrast_hist = args.contrast_hist
    noncontrast_hist = args.noncontrast_hist
    diffusion = create_gaussian_diffusion(config, timestep_respacing=False)
    model = create_score_model(config, image_level_cond,contrast_hist or noncontrast_hist , args.cond_on_lumen_mask )

    if args.continue_training:

        filename = args.modelfilename
    

        with bf.BlobFile(bf.join(logger.get_dir(), filename), "rb") as f:
            
            
            model.load_state_dict(
                th.load(f.name, map_location=th.device('cuda'))
            )
        model.to(th.device('cuda'))
        

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = th.device(config.device)
    model.to(device)

    logger.log(f"Model number of parameters {pytorch_total_params}")
    schedule_sampler = create_named_schedule_sampler(config.score_model.training.schedule_sampler, diffusion)

    input = args.input
    trans = args.trans
    

    filter_train = args.filter_train
    filter_val = args.filter_val

    logger.log("creating data loader...")
    train_loader = loader.get_data_loader(args.dataset, args.train_data_dir, config, input,  trans,filter_train, split_set='train', generator=True)
    val_loader = loader.get_data_loader(args.dataset, args.val_data_dir, config, input,  trans,filter_val, split_set='val', generator=False)
    time_load_end = time.time()
    time_load = time_load_end - time_load_start
    logger.log("data loaded: time ", str(time_load))
    logger.log("training...")
    
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        val_data = val_loader,
        batch_size=config.score_model.training.batch_size,
        lr=config.score_model.training.lr,
        ema_rate=config.score_model.training.ema_rate,
        log_interval=config.score_model.training.log_interval,
        save_interval=config.score_model.training.save_interval,
        val_interval=config.score_model.training.val_interval,
        use_fp16=config.score_model.training.use_fp16,
        fp16_scale_growth=config.score_model.training.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=config.score_model.training.weight_decay,
        lr_decay_steps=config.score_model.training.lr_decay_steps,
        lr_decay_factor=config.score_model.training.lr_decay_factor,
        iterations=config.score_model.training.iterations,
        num_samples=config.sampling.num_samples,
        num_input_channels=config.score_model.num_input_channels,
        image_size=config.score_model.image_size,
        clip_denoised=config.sampling.clip_denoised,
        use_ddim=False,
        device=device,
        args=args
    ).run_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    parser.add_argument("--dataset", help="brats", type=str, default='oxaaa')
    parser.add_argument("--input", help="input modality, choose from flair, t2, t1", type=str, default='noncontrast')
    parser.add_argument("--trans", help="translated modality, choose from flair, t2, t1", type=str, default='contrast')
    #parser.add_argument("--data_dir", help="data directory", type=str, default='/home/trin4156/Desktop/datasets/nnunet/nnunet_raw/Dataset102_nonconoxaaa2d/OxAAA')
    parser.add_argument("--train_data_dir", help="data directory", type=str, default='/mnt/data/data/OxAAA/train/normalized')
    parser.add_argument("--val_data_dir", help="data directory", type=str, default='/mnt/data/data/OxAAA/train/normalized')
    parser.add_argument("--experiment_name", help="model saving file name", type=str, default='None')
    parser.add_argument("--model_name", help="translated model: unet or diffusion", type=str, default='diffusion')
    parser.add_argument("--filter_train", help="a npy to filter data based on pixel difference and mask difference", type=str, default='/mnt/data/data/OxAAA/train/normalized/train_with_lumenmask_filtered.npy')
    parser.add_argument("--filter_val", help="a npy to filter data based on pixel difference and mask difference", type=str, default='/mnt/data/data/OxAAA/train/normalized/val_with_lumenmask_filtered.npy')
    parser.add_argument("--contrast_hist", help="a npy to filter data based on pixel difference and mask difference", action="store_true")
    parser.add_argument("--noncontrast_hist", help="a npy to filter data based on pixel difference and mask difference", action="store_true")
    parser.add_argument("--cond_on_noncontrast_mask", help="a npy to filter data based on pixel difference and mask difference", action="store_true")
    parser.add_argument("--cond_on_contrast_mask", help="a npy to filter data based on pixel difference and mask difference", action="store_true")
    parser.add_argument("--continue_training", help="a npy to filter data based on pixel difference and mask difference", action="store_true")
    parser.add_argument("--modelfilename", help="brats", type=str, default='model400000_cond_nonconarota_cond_nonconhist.pt')
    parser.add_argument("--continue_step", help="brats", type=str, default='400000')
    parser.add_argument("--cond_on_lumen_mask", help="brats",  action="store_true")
    parser.add_argument("--sdg_lumen_mask", help="brats",  action="store_true")
    

    args = parser.parse_args()
    main(args)


