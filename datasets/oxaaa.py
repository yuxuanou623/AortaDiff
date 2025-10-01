import os

import numpy as np
import monai.transforms as transforms
from os.path import join
from pathlib import Path
import nibabel as nib
from torch.utils.data import Dataset

from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion

def load_image(image_path):
    with Image.open(image_path)as img:
        return np.array(img, dtype=np.float32)
    
import numpy as np
from PIL import Image

def load_image_grey(image_path):
    # Load image, convert to grayscale, and convert to numpy array
    with Image.open(image_path).convert('L') as img:
        img_array = np.array(img, dtype=np.float32)
    
    # Normalize to range [0, 1]
    img_array /= 255.0
    
    # Scale to range [-1, 1]
    img_array = 2 * img_array - 1
    
    return img_array

# data_dict = {'contrast': trans_image, 'contrast_mask_tolerated':trans_mask_tolerated, 'noncon_arota':input_contrast, 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated}
#{ 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated, 'noncon_arota':input_contrast, 'input_img':input_img}
def get_oxaaa_base_transform_abnormalty_train_partial(image_size):
    base_transform = [
        transforms.AddChanneld(
            keys=['contrast',  'contrast_mask_tolerated','noncon_arota', 'noncontrast_mask_tolerated','square_mask']),
        transforms.Resized(
            keys=['contrast', 'noncon_arota'],
            spatial_size=(image_size, image_size)),
        transforms.Resized(
            keys=['contrast_mask_tolerated','noncontrast_mask_tolerated','square_mask',],
            spatial_size=(image_size, image_size),
            mode='nearest')
       
    ]

    return base_transform


def get_oxaaa_base_transform_abnormalty_test(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['contrast',  'contrast_mask_tolerated','noncon_arota', 'noncontrast_mask_tolerated','square_mask','trans_lumen_mask_tolerated', 'm_sdf', 'input_img']),
        transforms.Resized(
            keys=['contrast', 'noncon_arota', 'm_sdf', 'input_img'],
            spatial_size=(image_size, image_size)),
        transforms.Resized(
            keys=['contrast_mask_tolerated','noncontrast_mask_tolerated','square_mask','trans_lumen_mask_tolerated'],
            spatial_size=(image_size, image_size),
            mode='nearest')
       
    ]

    return base_transform
def get_oxaaa_base_transform_abnormalty_train(image_size):
    base_transform = [
        transforms.AddChanneld(
            keys=['contrast',  'contrast_mask_tolerated','noncon_arota', 'noncontrast_mask_tolerated','square_mask','trans_lumen_mask_tolerated', 'm_sdf']),
        transforms.Resized(
            keys=['contrast', 'noncon_arota', 'm_sdf'],
            spatial_size=(image_size, image_size)),
        transforms.Resized(
            keys=['contrast_mask_tolerated','noncontrast_mask_tolerated','square_mask','trans_lumen_mask_tolerated'],
            spatial_size=(image_size, image_size),
            mode='nearest')
       
    ]

    return base_transform


def get_oxaaa_train_transform_abnormalty_test(image_size):
    base_transform = get_oxaaa_base_transform_abnormalty_test(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['contrast', 'contrast_mask_tolerated', 'noncon_arota', 'trans_hist', 'input_hist', 'noncontrast_mask_tolerated','square_mask','trans_lumen_mask_tolerated', 'm_sdf','input_img']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_oxaaa_train_transform_abnormalty_train(image_size):
    base_transform = get_oxaaa_base_transform_abnormalty_train(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['contrast', 'contrast_mask_tolerated', 'noncon_arota', 'trans_hist', 'input_hist', 'noncontrast_mask_tolerated','square_mask','trans_lumen_mask_tolerated', 'm_sdf']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_oxaaa_train_transform_abnormalty_train_partial(image_size):
    base_transform = get_oxaaa_base_transform_abnormalty_train_partial(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['contrast', 'contrast_mask_tolerated', 'noncon_arota', 'trans_hist', 'input_hist', 'noncontrast_mask_tolerated','square_mask']),
    ]
    return transforms.Compose(base_transform + data_aug)







class OxAAADataset(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod='noncontrast', trans_mod='contrast',transforms=None, filter=None, mask_type = 'lumen'):
        super(OxAAADataset, self).__init__()
        assert mode in ['train', 'test', 'val'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.input_mod = input_mod  
        self.trans_mod = trans_mod  
        self.transforms = transforms
        self.mask_type = mask_type
        
        
        self.data_root =  Path(self.data_root) 
       
        # Initialize directories for contrast and non-contrast images
        self.input_dir = Path(self.data_root) / 'noncontrast'
        self.trans_dir = Path(self.data_root) / 'contrast'

        self.input_mask_dir = Path(self.data_root) / 'noncontrastmask'
        self.trans_mask_dir = Path(self.data_root) / 'contrastaortamask'
        if self.mask_type == 'lumen':
            self.trans_lumenmask_dir = Path(self.data_root) / 'contrastlumenmask'
        else:
            self.trans_lumenmask_dir = Path(self.data_root) / 'contrastthrombusmask'

        
        

    

        if filter is not None:
            self.filter = np.load(filter, allow_pickle=True)  # Load filenames from the npy file
            self.filter = set(self.filter.tolist())  # Convert to set for faster lookup
        else:
            self.filter = None
        

        # List of all image names in the input directory
        self.two_labels_input_images = sorted([img for img in self.input_dir.glob('*.nii.gz') if self.filter is None or img.name.split('/')[0] in self.filter])
        print(len(self.two_labels_input_images))
        
        


        # Dictionary to quickly find corresponding images
        self.image_pairs = self._cache_pairs()
        

    def _cache_pairs(self):
        pairs = []
        if self.mode == 'test':  # If mode is 'test', don't include the masks
            
            for input_img in self.two_labels_input_images:
                input_mask_path = self.input_mask_dir / input_img.name
                trans_img_path = self.trans_dir / input_img.name
                trans_mask_path = self.trans_mask_dir / input_img.name
                trans_lumen_mask_path = self.trans_lumenmask_dir / input_img.name

                
                
                if  input_mask_path.exists() and trans_img_path.exists() and trans_mask_path.exists() and trans_lumen_mask_path.exists():
                    pairs.append(( trans_img_path, trans_mask_path, input_img, input_mask_path, trans_lumen_mask_path))

        else:  # Otherwise, include masks
            # Pair images with the same name in input and trans directories
            for input_img in self.two_labels_input_images:
                trans_img_path = self.trans_dir / input_img.name
                
                input_mask_path = self.input_mask_dir / input_img.name
                trans_mask_path = self.trans_mask_dir / input_img.name
                
                trans_lumen_mask_path = self.trans_lumenmask_dir / input_img.name

               
                
                if trans_img_path.exists()  and trans_mask_path.exists() and input_mask_path.exists()   and trans_lumen_mask_path.exists():
                    pairs.append(( trans_img_path, trans_mask_path, input_img, input_mask_path, trans_lumen_mask_path))
        return pairs

    def __getitem__(self, index):
        # Depending on the mode, the pairs may have different numbers of elements
        if self.mode == 'test':
            trans_img_path,trans_mask_path , input_img_path, input_mask_path, trans_lumen_mask_path= self.image_pairs[index]
            # Load images

            trans_image = nib.load(trans_img_path).get_fdata()
           
            trans_mask = nib.load(trans_mask_path).get_fdata()
            input_img = nib.load(input_img_path).get_fdata()
            input_mask = nib.load(input_mask_path).get_fdata()
            trans_lumen_mask = nib.load(trans_lumen_mask_path).get_fdata()
            # coarse_mask_path = nib.load(coarse_mask_path).get_fdata()

            tolerance = 1e-3
            
            
            trans_mask_tolerated = (np.abs(trans_mask - 1) < tolerance).astype(int)
            input_mask_tolerated = (np.abs(input_mask - 1) < tolerance).astype(int)
            trans_lumen_mask_tolerated= (np.abs(trans_lumen_mask - 1) < tolerance).astype(int)
            #coarse_mask_path_tolerated= (np.abs(coarse_mask_path - 1) < tolerance).astype(int)
            y, x = np.where(input_mask_tolerated == 1)
            if len(x)==0:
                print("input_mask_path",input_mask_path)

            intersection_mask = np.logical_and(np.abs(input_mask - 1) < tolerance, np.abs(trans_mask - 1) < tolerance).astype(int)
            margin = 10  # Define how much bigger the square should be
            min_x = max(min(x) - margin, 0)
            max_x = min(max(x) + margin + 1, intersection_mask.shape[1])
            min_y = max(min(y) - margin, 0)
            max_y = min(max(y) + margin + 1, intersection_mask.shape[0])

            # Create a new mask with the square
            square_mask = np.zeros_like(intersection_mask)
            square_mask[min_y:max_y, min_x:max_x] = 1



            input_contrast = input_img*square_mask
            masked_pixels = trans_image[trans_mask_tolerated > 0]
            masked_tensor = torch.tensor(masked_pixels, dtype=torch.float32)  # Ensure it's float for histc

            trans_hist = torch.histc(masked_tensor, bins=32, min=-1, max=1) / trans_mask_tolerated.sum()
            masked_input = torch.tensor(input_img[input_mask_tolerated > 0], dtype=torch.float32)

            # Convert denominator if needed
            denominator = torch.tensor(input_mask_tolerated.sum()) if isinstance(input_mask_tolerated, np.ndarray) else input_mask_tolerated.sum()

            # Compute histogram
            input_hist = torch.histc(masked_input, bins=32, min=-1, max=1) / denominator


            lumen_bd = np.abs(binary_erosion(trans_lumen_mask_tolerated) - trans_lumen_mask_tolerated)
            distance = distance_transform_edt(np.where(lumen_bd== 0., np.ones_like(lumen_bd), np.zeros_like(lumen_bd)))
            m_sdf = np.where(trans_lumen_mask_tolerated == 1, distance * -1, distance)  # ensure signed DT

            # Truncate at threshold and normalize between [-1, 1]
            thresh = 15
            m_sdf = np.clip(m_sdf, -thresh, thresh)  # a cleaner way
            m_sdf /= thresh

            # coarse_lumen_bd = np.abs(binary_erosion(coarse_mask_path_tolerated) - coarse_mask_path_tolerated)
            # coarse_distance = distance_transform_edt(np.where(coarse_lumen_bd== 0., np.ones_like(coarse_lumen_bd), np.zeros_like(coarse_lumen_bd)))
            # coarse_m_sdf = np.where(coarse_mask_path_tolerated == 1, coarse_distance * -1, coarse_distance)  # ensure signed DT

            # Truncate at threshold and normalize between [-1, 1]
            thresh = 15
            # coarse_m_sdf = np.clip(coarse_m_sdf, -thresh, thresh)  # a cleaner way
            # coarse_m_sdf /= thresh

            


    
  

            data_dict = {'contrast': trans_image, 'contrast_mask_tolerated':trans_mask_tolerated, 'noncon_arota':input_contrast, 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated,'square_mask': square_mask, 'trans_lumen_mask_tolerated': trans_lumen_mask_tolerated, 'm_sdf':m_sdf, 'input_img': input_img}



        else:
            trans_img_path,trans_mask_path , input_img_path, input_mask_path, trans_lumen_mask_path= self.image_pairs[index]
            # Load images

            trans_image = nib.load(trans_img_path).get_fdata()
           
            trans_mask = nib.load(trans_mask_path).get_fdata()
            input_img = nib.load(input_img_path).get_fdata()
            input_mask = nib.load(input_mask_path).get_fdata()
            
            trans_lumen_mask = nib.load(trans_lumen_mask_path).get_fdata()
          
            

            tolerance = 1e-3
            
            
            trans_mask_tolerated = (np.abs(trans_mask - 1) < tolerance).astype(int)
            input_mask_tolerated = (np.abs(input_mask - 1) < tolerance).astype(int)
            


            
            
            y, x = np.where(input_mask_tolerated == 1)
            if len(x)==0:
                print("input_mask_path",input_mask_path)

            intersection_mask = np.logical_and(np.abs(input_mask - 1) < tolerance, np.abs(trans_mask - 1) < tolerance).astype(int)
            margin = 10  # Define how much bigger the square should be
            min_x = max(min(x) - margin, 0)
            max_x = min(max(x) + margin + 1, intersection_mask.shape[1])
            min_y = max(min(y) - margin, 0)
            max_y = min(max(y) + margin + 1, intersection_mask.shape[0])

            # Create a new mask with the square
            square_mask = np.zeros_like(intersection_mask)
            square_mask[min_y:max_y, min_x:max_x] = 1



            input_contrast = input_img*square_mask
            masked_pixels = trans_image[trans_mask_tolerated > 0]
            masked_tensor = torch.tensor(masked_pixels, dtype=torch.float32)  # Ensure it's float for histc

            trans_hist = torch.histc(masked_tensor, bins=32, min=-1, max=1) / trans_mask_tolerated.sum()
            masked_input = torch.tensor(input_img[input_mask_tolerated > 0], dtype=torch.float32)

            # Convert denominator if needed
            denominator = torch.tensor(input_mask_tolerated.sum()) if isinstance(input_mask_tolerated, np.ndarray) else input_mask_tolerated.sum()

            # Compute histogram
            input_hist = torch.histc(masked_input, bins=32, min=-1, max=1) / denominator


            

            # Truncate at threshold and normalize between [-1, 1]
            thresh = 15
            

            


            
            trans_lumen_mask = nib.load(trans_lumen_mask_path).get_fdata()
            trans_lumen_mask_tolerated= (np.abs(trans_lumen_mask - 1) < tolerance).astype(int)
            lumen_bd = np.abs(binary_erosion(trans_lumen_mask_tolerated) - trans_lumen_mask_tolerated)
            distance = distance_transform_edt(np.where(lumen_bd== 0., np.ones_like(lumen_bd), np.zeros_like(lumen_bd)))
            m_sdf = np.where(trans_lumen_mask_tolerated == 1, distance * -1, distance)  # ensure signed DT
            m_sdf = np.clip(m_sdf, -thresh, thresh)  # a cleaner way
            m_sdf /= thresh
            
            

            

            


    
  

            data_dict = {'contrast': trans_image, 'contrast_mask_tolerated':trans_mask_tolerated, 'noncon_arota':input_contrast, 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated,'square_mask': square_mask, 'trans_lumen_mask_tolerated': trans_lumen_mask_tolerated, 'm_sdf':m_sdf}



         
        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict

    def __len__(self):
        return len(self.image_pairs)


class OxAAADataset_partial(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod='noncontrast', trans_mod='contrast',transforms=None, filter=None):
        super(OxAAADataset_partial, self).__init__()
        assert mode in ['train', 'test', 'val'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.input_mod = input_mod  
        self.trans_mod = trans_mod  
        self.transforms = transforms
        
        
        self.data_root =  Path(self.data_root) 
       
        # Initialize directories for contrast and non-contrast images
        self.input_dir = Path(self.data_root) / 'noncontrast'
        self.trans_dir = Path(self.data_root) / 'contrast'

        self.input_mask_dir = Path(self.data_root) / 'noncontrastmask'
        self.trans_mask_dir = Path(self.data_root) / 'contrastaortamask'

        if filter is not None:
            self.filter = np.load(filter, allow_pickle=True)  # Load filenames from the npy file
            self.filter = set(self.filter.tolist())  # Convert to set for faster lookup
        else:
            self.filter = None

        # List of all image names in the input directory
        self.input_images = sorted([img for img in self.input_dir.glob('*.nii.gz') if self.filter is None or img.name.split('/')[0] in self.filter])
        


        # Dictionary to quickly find corresponding images
        self.image_pairs = self._cache_pairs()
        

    def _cache_pairs(self):
        pairs = []
        if self.mode == 'test':  # If mode is 'test', don't include the masks
            
            for input_img in self.input_images:
                input_mask_path = self.input_mask_dir / input_img.name
                trans_img_path = self.trans_dir / input_img.name
                trans_mask_path = self.trans_mask_dir / input_img.name
                trans_lumen_mask_path = self.trans_lumenmask_dir / input_img.name
                coarse_mask_path = self.coarse_mask_dir / input_img.name
                if  input_mask_path.exists() and trans_img_path.exists() and trans_mask_path.exists() and trans_lumen_mask_path.exists():
                    pairs.append(( trans_img_path, trans_mask_path, input_img, input_mask_path, trans_lumen_mask_path, coarse_mask_path))

        else:  # Otherwise, include masks
            # Pair images with the same name in input and trans directories
            for input_img in self.input_images:
                trans_img_path = self.trans_dir / input_img.name
                
                input_mask_path = self.input_mask_dir / input_img.name
                trans_mask_path = self.trans_mask_dir / input_img.name
                
                if trans_img_path.exists()  and trans_mask_path.exists() and input_mask_path.exists()  :
                    pairs.append(( trans_img_path, trans_mask_path, input_img, input_mask_path))
        return pairs

    def __getitem__(self, index):
        # Depending on the mode, the pairs may have different numbers of elements
        if self.mode == 'test':
            trans_img_path,trans_mask_path , input_img_path, input_mask_path, trans_lumen_mask_path= self.image_pairs[index]
            # Load images

            trans_image = nib.load(trans_img_path).get_fdata()
           
            trans_mask = nib.load(trans_mask_path).get_fdata()
            input_img = nib.load(input_img_path).get_fdata()
            input_mask = nib.load(input_mask_path).get_fdata()
            trans_lumen_mask = nib.load(trans_lumen_mask_path).get_fdata()
            

            tolerance = 1e-3
            
            
            trans_mask_tolerated = (np.abs(trans_mask - 1) < tolerance).astype(int)
            input_mask_tolerated = (np.abs(input_mask - 1) < tolerance).astype(int)
            trans_lumen_mask_tolerated= (np.abs(trans_lumen_mask - 1) < tolerance).astype(int)
            
            y, x = np.where(input_mask_tolerated == 1)
            if len(x)==0:
                print("input_mask_path",input_mask_path)

            intersection_mask = np.logical_and(np.abs(input_mask - 1) < tolerance, np.abs(trans_mask - 1) < tolerance).astype(int)
            margin = 10  # Define how much bigger the square should be
            min_x = max(min(x) - margin, 0)
            max_x = min(max(x) + margin + 1, intersection_mask.shape[1])
            min_y = max(min(y) - margin, 0)
            max_y = min(max(y) + margin + 1, intersection_mask.shape[0])

            # Create a new mask with the square
            square_mask = np.zeros_like(intersection_mask)
            square_mask[min_y:max_y, min_x:max_x] = 1



            input_contrast = input_img*square_mask
            masked_pixels = trans_image[trans_mask_tolerated > 0]
            masked_tensor = torch.tensor(masked_pixels, dtype=torch.float32)  # Ensure it's float for histc

            trans_hist = torch.histc(masked_tensor, bins=32, min=-1, max=1) / trans_mask_tolerated.sum()
            masked_input = torch.tensor(input_img[input_mask_tolerated > 0], dtype=torch.float32)

            # Convert denominator if needed
            denominator = torch.tensor(input_mask_tolerated.sum()) if isinstance(input_mask_tolerated, np.ndarray) else input_mask_tolerated.sum()

            # Compute histogram
            input_hist = torch.histc(masked_input, bins=32, min=-1, max=1) / denominator


            lumen_bd = np.abs(binary_erosion(trans_lumen_mask_tolerated) - trans_lumen_mask_tolerated)
            distance = distance_transform_edt(np.where(lumen_bd== 0., np.ones_like(lumen_bd), np.zeros_like(lumen_bd)))
            m_sdf = np.where(trans_lumen_mask_tolerated == 1, distance * -1, distance)  # ensure signed DT

            # Truncate at threshold and normalize between [-1, 1]
            thresh = 15
            m_sdf = np.clip(m_sdf, -thresh, thresh)  # a cleaner way
            m_sdf /= thresh

            

            # Truncate at threshold and normalize between [-1, 1]
            thresh = 15

            data_dict = {'contrast': trans_image, 'contrast_mask_tolerated':trans_mask_tolerated, 'noncon_arota':input_contrast, 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated,'square_mask': square_mask, 'trans_lumen_mask_tolerated': trans_lumen_mask_tolerated, 'm_sdf':m_sdf, 'input_img': input_img}



        else:
            trans_img_path,trans_mask_path , input_img_path, input_mask_path= self.image_pairs[index]
            # Load images

            trans_image = nib.load(trans_img_path).get_fdata()
           
            trans_mask = nib.load(trans_mask_path).get_fdata()
            input_img = nib.load(input_img_path).get_fdata()
            input_mask = nib.load(input_mask_path).get_fdata()
            

            tolerance = 1e-3
            
            
            trans_mask_tolerated = (np.abs(trans_mask - 1) < tolerance).astype(int)
            input_mask_tolerated = (np.abs(input_mask - 1) < tolerance).astype(int)
            


            
            
            y, x = np.where(input_mask_tolerated == 1)
            if len(x)==0:
                print("input_mask_path",input_mask_path)

            intersection_mask = np.logical_and(np.abs(input_mask - 1) < tolerance, np.abs(trans_mask - 1) < tolerance).astype(int)
            margin = 10  # Define how much bigger the square should be
            min_x = max(min(x) - margin, 0)
            max_x = min(max(x) + margin + 1, intersection_mask.shape[1])
            min_y = max(min(y) - margin, 0)
            max_y = min(max(y) + margin + 1, intersection_mask.shape[0])

            # Create a new mask with the square
            square_mask = np.zeros_like(intersection_mask)
            square_mask[min_y:max_y, min_x:max_x] = 1



            input_contrast = input_img*square_mask
            masked_pixels = trans_image[trans_mask_tolerated > 0]
            masked_tensor = torch.tensor(masked_pixels, dtype=torch.float32)  # Ensure it's float for histc

            trans_hist = torch.histc(masked_tensor, bins=32, min=-1, max=1) / trans_mask_tolerated.sum()
            masked_input = torch.tensor(input_img[input_mask_tolerated > 0], dtype=torch.float32)

            # Convert denominator if needed
            denominator = torch.tensor(input_mask_tolerated.sum()) if isinstance(input_mask_tolerated, np.ndarray) else input_mask_tolerated.sum()

            # Compute histogram
            input_hist = torch.histc(masked_input, bins=32, min=-1, max=1) / denominator

            # Truncate at threshold and normalize between [-1, 1]
            thresh = 15

            data_dict = {'contrast': trans_image, 'contrast_mask_tolerated':trans_mask_tolerated, 'noncon_arota':input_contrast, 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated,'square_mask': square_mask}



         
        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict

    def __len__(self):
        return len(self.image_pairs)