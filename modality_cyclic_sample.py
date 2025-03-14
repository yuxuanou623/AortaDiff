"""
Like score_sampling.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import os
import sys
import argparse
import random
import cv2
from skimage import metrics
import numpy as np
import torch as th
import blobfile as bf
from pathlib import Path
from skimage import morphology
from sklearn.metrics import roc_auc_score, jaccard_score

from datasets import loader
from configs import get_config
from utils import logger
from utils.script_util import create_gaussian_diffusion, create_score_model
from utils.binary_metrics import assd_metric, sensitivity_metric, precision_metric
sys.path.append(str(Path.cwd()))
from tqdm import tqdm
from PIL import Image


def normalize(img, _min=None, _max=None):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0.5)
    pred = pred.astype(int)
    targs = targs.astype(int)
    return 2. * (pred*targs).sum() / (pred+targs).sum()

import os
import cv2
import numpy as np
import torch  # If working with PyTorch tensors

import os
import numpy as np
import cv2
import torch

def save_images_and_calculate_metrics(img_pred_all, img_true_all, trans_all, output_folder, n):
    """
    Saves the first `n` images from img_pred_all, img_true_all, and trans_all as a single PNG file with
    each image side by side, and calculates average PSNR and SSIM for pred vs true and pred vs trans.

    Parameters:
    - img_pred_all: (numpy array or torch.Tensor) Predicted images (bs, 1, 512, 512).
    - img_true_all: (numpy array or torch.Tensor) Ground truth images (bs, 1, 512, 512).
    - trans_all: (numpy array or torch.Tensor) Transformed images (bs, 1, 512, 512).
    - output_folder: (str) Folder to save combined images.
    - n: (int) Number of images to save.
    
    Returns:
    - Average PSNR between predicted and true images.
    - Average SSIM between predicted and true images.
    - Average SSIM between predicted and transformed images.
    """

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert tensors to NumPy if needed
    if isinstance(img_pred_all, torch.Tensor):
        img_pred_all = img_pred_all.cpu().numpy()
    if isinstance(img_true_all, torch.Tensor):
        img_true_all = img_true_all.cpu().numpy()
    if isinstance(trans_all, torch.Tensor):
        trans_all = trans_all.cpu().numpy()

    # Ensure `n` does not exceed available images
    n = min(n, img_pred_all.shape[0], img_true_all.shape[0], trans_all.shape[0])

    psnr_values = []
    ssim_pred_true_values = []
    ssim_pred_trans_values = []

    # Loop through the first `n` images
    for i in range(n):
        # Normalize and convert images to [0, 255]
        def normalize_image(image):
            image = np.squeeze(image)  # Remove channel dimension
            image = ((image + 1) / 2) * 255  # Convert from [-1, 1] to [0, 255]
            return image.astype(np.uint8)
        
        pred = normalize_image(img_pred_all[i])
        true = normalize_image(img_true_all[i])
        trans = normalize_image(trans_all[i])

        # Calculate PSNR and SSIM
        psnr_pred_true = metrics.peak_signal_noise_ratio(true, pred)
        ssim_pred_true = metrics.structural_similarity(true, pred)
        ssim_pred_trans = metrics.structural_similarity(trans, pred)

        psnr_values.append(psnr_pred_true)
        ssim_pred_true_values.append(ssim_pred_true)
        ssim_pred_trans_values.append(ssim_pred_trans)

        # Stack images horizontally
        combined_image = np.hstack((true, trans, pred))

        # Save the combined image
        filename = os.path.join(output_folder, f"combined_image_{i}.png")
        cv2.imwrite(filename, combined_image)

    # Compute average PSNR and SSIM
    average_psnr = np.mean(psnr_values)
    average_ssim_pred_true = np.mean(ssim_pred_true_values)
    average_ssim_pred_trans = np.mean(ssim_pred_trans_values)

    print(f"Average PSNR (Pred vs True): {average_psnr:.2f} dB")
    print(f"Average SSIM (Pred vs True): {average_ssim_pred_true:.4f}")
    print(f"Average SSIM (Pred vs Trans): {average_ssim_pred_trans:.4f}")

    return average_psnr, average_ssim_pred_true, average_ssim_pred_trans




def main(args):
    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)
    config = get_config.file_from_dataset(args.dataset)
    if args.experiment_name_forward != 'None':
        experiment_name = args.experiment_name_forward
    else:
        raise Exception("Experiment name does exit")
    logger.configure(Path(experiment_name) / "score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])
    logger.configure(Path(experiment_name) / "score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating loader...")

    test_loader = loader.get_data_loader(args.dataset, args.data_dir, config, args.input, args.trans, args.filter, split_set='test',
                                          generator=False)
    

        # Define the folder to save images
    
    


    logger.log("creating model and diffusion...")
    if args.model_name == 'unet':
        image_level_cond_forward = False
        image_level_cond_backward = False
    elif args.model_name == 'diffusion':
        image_level_cond_forward = True
        image_level_cond_backward = False
    else:
        raise Exception("Model name does exit")
    diffusion = create_gaussian_diffusion(config, args.timestep_respacing)
    model_forward = create_score_model(config, image_level_cond_forward)
    model_backward = create_score_model(config, image_level_cond_backward)

    filename = args.modelfilename
 

    with bf.BlobFile(bf.join(logger.get_dir(), filename), "rb") as f:
        
        
        model_forward.load_state_dict(
            th.load(f.name, map_location=th.device('cuda'))
        )
    model_forward.to(th.device('cuda'))

    experiment_name_backward = f.name.split(experiment_name)[0] + args.experiment_name_backward + f.name.split(experiment_name)[1]
    # model_backward.load_state_dict(
    #     th.load(experiment_name_backward, map_location=th.device('cuda'))
    # )
    # model_backward.to(th.device('cuda'))

    if config.score_model.use_fp16:
        model_forward.convert_to_fp16()
        # model_backward.convert_to_fp16()

    model_forward.eval() 
    # model_backward.eval()

    logger.log("sampling...")

    dice = np.zeros(100)
    auc = np.zeros(1)
    assd = np.zeros(1)
    sensitivity = np.zeros(1)
    precision = np.zeros(1)
    jaccard = np.zeros(1)

    num_batch = 0
    num_sample = 0
    
    n=2000
    img_true_all = np.zeros((n*(config.sampling.batch_size), config.score_model.num_input_channels, config.score_model.image_size,
             config.score_model.image_size))
    img_pred_all = np.zeros((n*(config.sampling.batch_size), config.score_model.num_input_channels, config.score_model.image_size,
            config.score_model.image_size))
    trans_all = np.zeros((n*(config.sampling.batch_size), config.score_model.num_input_channels, config.score_model.image_size,
             config.score_model.image_size))
    # brain_mask_all = np.zeros((len(test_loader.dataset), config.score_model.num_input_channels, config.score_model.image_size, config.score_model.image_size))
    # test_data_seg_all = np.zeros((len(test_loader.dataset), config.score_model.num_input_channels,
    #                            config.score_model.image_size, config.score_model.image_size))
    for i, test_data_dict in tqdm(enumerate(test_loader), total=len(test_loader)):
        if i >=n:
            break
        model_kwargs = {}
        ### brats dataset ###
        if args.dataset == 'brats':
            test_data_input = test_data_dict[1].pop('input').cuda()
            test_data_seg = test_data_dict[1].pop('seg')
            brain_mask = test_data_dict[1].pop('brainmask')
            brain_mask = (th.ones(brain_mask.shape) * (brain_mask > 0)).cuda()
            test_data_seg = (th.ones(test_data_seg.shape) * (test_data_seg > 0)).cuda()
        
        if args.dataset == 'ldfdct':
            
            
            # test_data_input = test_data_dict[1].pop('input').cuda()
            test_data_input = test_data_dict.pop('input').cuda()
            # test_data_seg = test_data_dict[1].pop('trans')
            test_data_seg = test_data_dict.pop('trans')

        elif args.dataset == 'oxaaa':
            
            
            # test_data_input = test_data_dict[1].pop('input').cuda()
            test_data_input = test_data_dict.pop('input').cuda()
            # test_data_seg = test_data_dict[1].pop('trans')
            test_data_seg = test_data_dict.pop('trans')

        sample_fn = (
                        diffusion.p_sample_loop
        )
        print("test_data_input",test_data_input.shape)
        
        # sample = sliding_window_inference(test_data_input,sample_fn,  model_forward, model_backward, model_kwargs, config, args,
        #                      patch_size=128, stride=128, batch_size=32 )
        
        sample = sample_fn(
            model_forward, model_backward, test_data_input, num_batch,
            (test_data_seg.shape[0], config.score_model.num_input_channels, config.score_model.image_size,
             config.score_model.image_size),
            model_name=args.model_name,
            clip_denoised=config.sampling.clip_denoised,  # is True, clip the denoised signal into [-1, 1].
            model_kwargs=model_kwargs,  # reconstruction = True
            eta=config.sampling.eta,
            model_forward_name=args.experiment_name_forward,
            model_backward_name=args.experiment_name_backward,
            ddim=args.use_ddim

        )
        num_batch += 1
        sample_datach = sample.detach().cpu().numpy()
        print(sample_datach.max())
        print(sample_datach.min())
        print("sample_datach", sample_datach.shape)
        test_data_seg_detach = test_data_seg.detach().cpu().numpy()
        print("test_data_input_detach", test_data_seg_detach.shape)

        print("test_data_input.shape[0]",test_data_input.shape[0])
        print("test_data_input.detach().cpu().numpy()",test_data_input.detach().cpu().numpy().shape)
        img_true_all[num_sample:num_sample+test_data_input.shape[0]] = test_data_input.detach().cpu().numpy()
        img_pred_all[num_sample:num_sample+test_data_input.shape[0]] = sample.detach().cpu().numpy()
        trans_all[num_sample:num_sample+test_data_input.shape[0]]=test_data_seg.detach().cpu().numpy()

        num_sample += test_data_input.shape[0]
    logger.log("all the confidence maps from the testing set saved...")
    if args.model_name == 'diffusion':
        print("img_pred_all", img_pred_all.shape)
        error = (trans_all - img_pred_all) ** 2
        print("error", type(error))
        def mean_flat(tensor):
            """
            Take the mean over all non-batch dimensions.
            """
            return tensor.mean(axis=tuple(range(1, len(tensor.shape))))
        print("meanflat",mean_flat(error))
        print("meanflat mea ",np.mean(mean_flat(error)))

        error = np.array(error)
        print("sum",error.sum())
        error_map = normalize(error)
        print("error_map")
        print(type(error_map))
        print("error_map",error_map.sum())
        output_folder_pred = "/mnt/data/data/evaluation/predict" +filename[:-3] + "timestep1000"# Change to your actual folder
      
        # save_images(img_pred_all, img_true_all, trans_all,output_folder_pred, output_folder_true,output_folder_trans,num_sample)
        save_images_and_calculate_metrics(img_pred_all, img_true_all, trans_all,output_folder_pred, num_sample)
    elif args.model_name == 'diffusion_':
        filename_mask = "mask_forward_"+args.experiment_name_forward+'_backward_'+args.experiment_name_backward+".pt"
        filename_x0 = "cyclic_predict_"+args.experiment_name_forward+'_backward_'+args.experiment_name_backward+".pt"
        with bf.BlobFile(bf.join(logger.get_dir(), filename_mask), "rb") as f:
            tensor_load_mask = th.load(f)
        with bf.BlobFile(bf.join(logger.get_dir(), filename_x0), "rb") as f:
            tensor_load_xpred = th.load(f)
        load_gt_repeat = np.expand_dims(img_true_all, axis=0).repeat(tensor_load_mask.shape[0], axis=0)
        error_map = (np.abs(tensor_load_xpred.numpy() - load_gt_repeat)) ** 2
        mean_error_map = np.sum(error_map * (1-tensor_load_mask.numpy()), 0) / np.sum((1-tensor_load_mask.numpy()), 0)
        error_map = normalize(np.where(np.isnan(mean_error_map), 0, mean_error_map))
    
def reseed_random(seed):
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    parser.add_argument("--dataset", help="brats", type=str, default='oxaaa')
    parser.add_argument("--input", help="input modality, choose from flair, t2, t1", type=str, default='noncon')
    parser.add_argument("--trans", help="input modality, choose from flair, t2, t1", type=str, default='con')
    parser.add_argument("--data_dir", help="data directory", type=str, default='/mnt/data/data/OxAAA/test/normalized')
    parser.add_argument("--experiment_name_forward", help="forward model saving file name", type=str, default='diffusion_oxaaa_noncon_con')
    parser.add_argument("--experiment_name_backward", help="backward model saving file name", type=str, default='meiyou')
    parser.add_argument("--model_name", help="translated model: unet or diffusion", type=str, default='diffusion')
    parser.add_argument("--use_ddim", help="if you want to use ddim during sampling, True or False", type=str, default='True')
    parser.add_argument("--timestep_respacing", help="If you want to rescale timestep during sampling. enter the timestep you want to rescale the diffusion prcess to. If you do not wish to resale thetimestep, leave it blank or put 1000.", type=int,
                        default=1000)
    parser.add_argument("--modelfilename", help="brats", type=str, default='model155000_batchsize32_filtereddata.pt')
    parser.add_argument("--filter", help="a npy to filter data based on pixel difference and mask difference", type=str, default=None)

    args = parser.parse_args()
    print(args.dataset)
    main(args)
