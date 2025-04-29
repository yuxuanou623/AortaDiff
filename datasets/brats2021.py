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

def get_oxaaa_base_transform_abnormalty_test(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input_img','noncon_arota', 'noncontrast_mask_tolerated', 'trans_image','contrast_mask_tolerated','trans_lumen_mask_tolerated']),
        transforms.Resized(
            keys=['noncon_arota', 'input_img', 'trans_image'],
            spatial_size=(image_size, image_size)),
        transforms.Resized(
            keys=['noncontrast_mask_tolerated','contrast_mask_tolerated','trans_lumen_mask_tolerated'],
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
            keys=['trans_hist', 'input_hist', 'noncontrast_mask_tolerated', 'noncon_arota', 'input_img','trans_image', 'contrast_mask_tolerated']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_oxaaa_train_transform_abnormalty_train(image_size):
    base_transform = get_oxaaa_base_transform_abnormalty_train(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['contrast', 'contrast_mask_tolerated', 'noncon_arota', 'trans_hist', 'input_hist', 'noncontrast_mask_tolerated','square_mask','trans_lumen_mask_tolerated', 'm_sdf']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_ldfdct_train_transform_abnormalty_train(image_size):
    base_transform = get_ldfdct_base_transform_abnormalty_train(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'trans']),
    ]
    return transforms.Compose(base_transform + data_aug)
def get_ldfdct_base_transform_abnormalty_train(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'trans']),
        transforms.Resized(
            keys=['input', 'trans'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform

def get_ldfdct_train_transform_abnormalty_test(image_size):
    base_transform = get_ldfdct_base_transform_abnormalty_test(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'trans']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_brats2021_train_transform_abnormalty_test(image_size):
    base_transform = get_brats2021_base_transform_abnormalty_test(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'brainmask', 'seg']),
    ]
    return transforms.Compose(base_transform + data_aug)
def get_ldfdct_base_transform_abnormalty_test(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'trans']),
        transforms.Resized(
            keys=['input', 'trans'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform

def get_brats2021_base_transform_abnormalty_test(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'brainmask', 'seg']),
        transforms.Resized(
            keys=['input', 'brainmask', 'seg'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform

class BraTS2021Dataset_Cyclic(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod, trans_mod=None, transforms=None):
        super(BraTS2021Dataset_Cyclic, self).__init__()

        assert mode in ['train', 'test', 'val'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.input_mod = input_mod

        self.transforms = transforms
        self.case_names_input = sorted(list(Path(os.path.join(self.data_root, input_mod)).iterdir()))
        self.case_names_brainmask = sorted(list(Path(os.path.join(self.data_root, 'brainmask')).iterdir()))
        self.case_names_seg = sorted(list(Path(os.path.join(self.data_root, 'seg')).iterdir()))
        if mode == 'train':
            self.trans_mod = trans_mod
            self.case_names_trans = sorted(list(Path(os.path.join(self.data_root, trans_mod)).iterdir()))

    def __getitem__(self, index: int) -> tuple:
        name_input = self.case_names_input[index].name
        name_brainmask = self.case_names_brainmask[index].name
        name_seg = self.case_names_seg[index].name
        base_dir_input = join(self.data_root, self.input_mod, name_input)
        base_dir_brainmask = join(self.data_root, 'brainmask', name_brainmask)
        base_dir_seg = join(self.data_root, 'seg', name_seg)
        input = np.load(base_dir_input).astype(np.float32)

        brain_mask = np.load(base_dir_brainmask).astype(np.float32)
        seg = np.load(base_dir_seg).astype(np.float32)
        if self.mode == 'train':
            name_trans = self.case_names_trans[index].name
            base_dir_trans = join(self.data_root, self.trans_mod, name_trans)
            trans = np.load(base_dir_trans).astype(np.float32)
            item = self.transforms(
                {'input': input, 'trans': trans, 'brainmask': brain_mask, 'seg': seg})
        else:
            item = self.transforms(
                {'input': input, 'brainmask': brain_mask, 'seg': seg})

        return item

    def __len__(self):
        return len(self.case_names_input)
    


class LDFDCTDataset(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod='ld', trans_mod='fd', transforms=None):
        super(LDFDCTDataset, self).__init__()
        assert mode in ['train', 'test'], 'Unknown mode'
        self.mode = mode
        folder_name = 'LD_FD_CT_train' if self.mode == 'train' else 'LD_FD_CT_test'
        self.data_root = os.path.join(data_root, folder_name)
        self.input_mod = input_mod
        self.trans_mod = trans_mod
        self.transforms = transforms
        
        # Gather all sample directories
        self.sample_dirs = sorted(
            [d for d in Path(self.data_root).iterdir() 
             if d.is_dir() and not d.name.startswith('.') and 'ipynb_checkpoints' not in d.name]
        )
        
        self.dir_index = 0  # Index to track the current directory
        self.pair_index = 0  # Index to track the current pair within the directory

        # Cache pairs per directory
        self.pairs_cache = []
        self._cache_pairs()
       

    def _cache_pairs(self):
        # Cache all image pairs from the current directory
        sample_dir = self.sample_dirs[self.dir_index]
        # print("sample_dir", sample_dir)
        
        
        image_files = list(sample_dir.glob("*.png"))
        
        
        input_images = [img for img in image_files if self.input_mod in img.name]
        print("self.trans_mod",self.trans_mod)
        trans_images = [img for img in image_files if self.trans_mod in img.name]
        # print("input_images", len(input_images))
        # print("trans_images", len(trans_images))
        
        for input_img in input_images:
            identifier = input_img.stem.split(f"_{self.input_mod}")[0]
            trans_img = next((img for img in trans_images if img.stem.startswith(identifier)), None)
            if trans_img:
                self.pairs_cache.append((input_img, trans_img))

        

    def __getitem__(self, index: int) -> dict:
        # Check if we need to refresh the cache
        if self.pair_index >= len(self.pairs_cache):
            self.dir_index =(self.dir_index + 1) % len(self.sample_dirs)
            self._cache_pairs()

        # Get the current pair
        # print("self.pairs_cache", len(self.pairs_cache))
        # print("self.pair_index", self.pair_index)
        input_img_path, trans_img_path = self.pairs_cache[self.pair_index]
        self.pair_index += 1
        
        # Load images
        input_image = load_image(input_img_path)
        trans_image = load_image(trans_img_path)
        print("input_image", input_image.shape)
        print("trans_image", trans_image.shape)
        data_dict = {'input': input_image, 'trans': trans_image}
        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict

    def __len__(self):
        return sum(len(list(Path(dir).glob("*.png"))) // 2 for dir in self.sample_dirs)



class OxAAADataset(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod='noncontrast', trans_mod='contrast',transforms=None, filter=None):
        super(OxAAADataset, self).__init__()
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
        self.trans_mask_dir = Path(self.data_root) / 'contrastmask'
        self.trans_lumenmask_dir = Path(self.data_root) / 'contrastlumenmask'

    

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
            # # Pair images in test mode (no masks)
            # for input_img in self.input_images:
            #     input_mask_path = self.input_mask_dir / input_img.name
            #     if input_mask_path.exists():
            #         pairs.append((input_img, input_mask_path))
            for input_img in self.input_images:
                input_mask_path = self.input_mask_dir / input_img.name
                trans_img_path = self.trans_dir / input_img.name
                trans_mask_path = self.trans_mask_dir / input_img.name
               
                if  input_mask_path.exists() and trans_img_path.exists() and trans_mask_path.exists():
                    pairs.append(( input_img, input_mask_path, trans_img_path, trans_mask_path))

        else:  # Otherwise, include masks
            # Pair images with the same name in input and trans directories
            for input_img in self.input_images:
                trans_img_path = self.trans_dir / input_img.name
               
                input_mask_path = self.input_mask_dir / input_img.name
                trans_mask_path = self.trans_mask_dir / input_img.name
                trans_lumen_mask_path = self.trans_lumenmask_dir / input_img.name
                if trans_img_path.exists()  and trans_mask_path.exists() and input_mask_path.exists() and trans_lumen_mask_path.exists():
                    pairs.append(( trans_img_path, trans_mask_path, input_img, input_mask_path, trans_lumen_mask_path))
        return pairs

    def __getitem__(self, index):
        # Depending on the mode, the pairs may have different numbers of elements
        if self.mode == 'test':
            input_img_path, input_mask_path, trans_img_path,trans_mask_path = self.image_pairs[index]
            # Load images

            trans_image = nib.load(trans_img_path).get_fdata()
           
            trans_mask = nib.load(trans_mask_path).get_fdata()
            input_img = nib.load(input_img_path).get_fdata()
            input_mask = nib.load(input_mask_path).get_fdata()

            tolerance = 1e-3
            trans_mask_tolerated = (np.abs(trans_mask - 1) < tolerance).astype(int)
            input_mask_tolerated = (np.abs(input_mask - 1) < tolerance).astype(int)
            input_contrast = input_mask_tolerated*input_img
            trans_contrast = trans_mask_tolerated*trans_image
            input_background = (1 - input_mask_tolerated)*input_img
            masked_pixels = trans_image[trans_mask_tolerated > 0]
            masked_tensor = torch.tensor(masked_pixels, dtype=torch.float32)  # Ensure it's float for histc

            trans_hist = torch.histc(masked_tensor, bins=32, min=-1, max=1) / trans_mask_tolerated.sum()
            masked_input = torch.tensor(input_img[input_mask_tolerated > 0], dtype=torch.float32)

            # Convert denominator if needed
            denominator = torch.tensor(input_mask_tolerated.sum()) if isinstance(input_mask_tolerated, np.ndarray) else input_mask_tolerated.sum()

            # Compute histogram
            input_hist = torch.histc(masked_input, bins=32, min=-1, max=1) / denominator

    
  

            data_dict = { 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated, 'noncon_arota':input_contrast, 'input_img':input_img, 'trans_image': trans_image, 'contrast_mask_tolerated': trans_mask_tolerated}



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

            


            # import matplotlib.pyplot as plt
        

            # # Assuming input_hist and trans_hist are PyTorch tensors of shape [4, 32]
            # # Convert them to NumPy arrays for plotting
            # input_hist_np = input_hist.cpu().numpy()
            # trans_hist_np = trans_hist.cpu().numpy()
            # print("trans_hist_np",trans_hist_np.shape)
            # print("input_hist_np",input_hist_np.shape)

            # # Assuming input_hist_np and trans_hist_np are NumPy arrays of shape (32,)
            # # Define the number of bins
            # num_bins = input_hist_np.shape[0]

            # # Create an array of bin centers
            # bin_edges = np.linspace(-1, 1, num_bins + 1)  # Assuming the histogram range is [-1, 1]
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # # Create the plot
            # plt.figure(figsize=(10, 6))
            # plt.plot(bin_centers, input_hist_np, label='Input Histogram', color='blue', marker='o')
            # plt.plot(bin_centers, trans_hist_np, label='Transformed Histogram', color='orange', marker='x')
            # plt.title('Comparison of Input and Transformed Histograms')
            # plt.xlabel('Intensity Bin Center')
            # plt.ylabel('Normalized Frequency')
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()

            # # Save the figure as a PNG file
            # plt.savefig('histogram_comparison.png')
            # plt.close()
            # import sys
            # sys.exit()


            
            # Create the tolerated mask for input_mask using the same method
    
  

            data_dict = {'contrast': trans_image, 'contrast_mask_tolerated':trans_mask_tolerated, 'noncon_arota':input_contrast, 'trans_hist':trans_hist,'input_hist':input_hist , 'noncontrast_mask_tolerated': input_mask_tolerated,'square_mask': square_mask, 'trans_lumen_mask_tolerated': trans_lumen_mask_tolerated, 'm_sdf':m_sdf}



         
        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict

    def __len__(self):
        return len(self.image_pairs)

# class OxAAADataset(Dataset):
#     def __init__(self, data_root: str, mode: str, input_mod='noncon', trans_mod='con', transforms=None):
#         super(OxAAADataset, self).__init__()
#         assert mode in ['train', 'test'], 'Unknown mode'
#         self.mode = mode
#         self.data_root = data_root
#         self.input_mod = input_mod  # Typically 'noncon'
#         self.trans_mod = trans_mod  # Typically 'con'
#         self.transforms = transforms
        
#         self.data_root =  Path(self.data_root) 
       
#         # Initialize directories for contrast and non-contrast images
#         self.input_dir = Path(self.data_root) / 'noncontrast'
#         self.trans_dir = Path(self.data_root) / 'contrast'

#         print("self.input_dir", self.input_dir)
#         print("self.trans_dir ", self.trans_dir )

#         # List of all image names in the input directory
#         self.input_images = sorted(self.input_dir.glob('*.nii.gz'))

#         # Dictionary to quickly find corresponding images
#         self.image_pairs = self._cache_pairs()

#     def _cache_pairs(self):
#         pairs = {}
#         # Pair images with the same name in input and trans directories
#         for input_img in self.input_images:
#             trans_img_path = self.trans_dir / input_img.name
#             if trans_img_path.exists():
#                 pairs[input_img] = trans_img_path
#         return pairs

#     def __getitem__(self, index):
#         input_img_path, trans_img_path = list(self.image_pairs.items())[index]
#         # Load images
#         input_image = nib.load(input_img_path).get_fdata()
#         trans_image = nib.load(trans_img_path).get_fdata()

#         data_dict = {'input': input_image, 'trans': trans_image}
#         # print("input_image", input_image.shape)
#         # print("trans_image", trans_image.shape)
#         if isinstance(data_dict['input'], bytes) or isinstance(data_dict['trans'], bytes):
#             print("input_img_path",input_img_path)
#             print("trans_img_path",trans_img_path)
#             raise ValueError("Image data is in bytes, expected a numerical array or tensor.")
#         if self.transforms:
#             data_dict = self.transforms(data_dict)

#         return data_dict

#     def __len__(self):
#         return len(self.image_pairs)
    
# class OxAAADataset(Dataset):
#     def __init__(self, data_root: str, mode: str, input_mod='noncon', trans_mod='con', transforms=None):
#         super(OxAAADataset, self).__init__()
#         assert mode in ['train', 'test'], 'Unknown mode'
#         self.mode = mode
#         self.data_root = data_root
#         self.input_mod = input_mod  # Typically 'noncon'
#         self.trans_mod = trans_mod  # Typically 'con'
#         self.transforms = transforms
        
#         self.data_root =  Path(self.data_root) / self.mode
       
#         # Initialize directories for contrast and non-contrast images
#         self.input_dir = Path(self.data_root) / 'noncon'
#         self.trans_dir = Path(self.data_root) / 'con'

#         print("self.input_dir", self.input_dir)
#         print("self.trans_dir ", self.trans_dir )

#         # List of all image names in the input directory
#         self.input_images = sorted(self.input_dir.glob('*.png'))

#         # Dictionary to quickly find corresponding images
#         self.image_pairs = self._cache_pairs()

#     def _cache_pairs(self):
#         pairs = {}
#         # Pair images with the same name in input and trans directories
#         for input_img in self.input_images:
#             trans_img_path = self.trans_dir / input_img.name
#             if trans_img_path.exists():
#                 pairs[input_img] = trans_img_path
#         return pairs

#     def __getitem__(self, index):
#         input_img_path, trans_img_path = list(self.image_pairs.items())[index]
#         # Load images
#         input_image = load_image_grey(input_img_path)
#         trans_image = load_image_grey(trans_img_path)

#         data_dict = {'input': input_image, 'trans': trans_image}
#         # print("input_image", input_image.shape)
#         # print("trans_image", trans_image.shape)
#         if isinstance(data_dict['input'], bytes) or isinstance(data_dict['trans'], bytes):
#             print("input_img_path",input_img_path)
#             print("trans_img_path",trans_img_path)
#             raise ValueError("Image data is in bytes, expected a numerical array or tensor.")
#         if self.transforms:
#             data_dict = self.transforms(data_dict)

#         return data_dict

#     def __len__(self):
#         return len(self.image_pairs)




# import matplotlib.pyplot as plt


# # Initialize the dataset
# data_root = '/home/trin4156/Downloads/data'  # Replace with the actual path to your data
# dataset = LDFDCTDataset(data_root=data_root, mode='train')
# # Initialize the dataset
# data_root = '/home/trin4156/Desktop/datasets/nnunet/nnunet_raw/Dataset102_nonconoxaaa2d/OxAAA'
# dataset = OxAAADataset(data_root=data_root, mode='train', input_mod='noncon', trans_mod='con')

# # Fetch a single batch (let's assume we define batch size as the number of pairs in one directory)
# batch_size = len(dataset._cache_pairs())
# batch_data = [dataset[i] for i in range(2)condrecontructcontrast]

# # Extract input and trans images' data for analysis
# input_images = np.array([data['input'].flatten() for data in batch_data])
# trans_images = np.array([data['trans'].flatten() for data in batch_data])

# # Plotting the data distribution
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.hist(input_images.flatten(), bins=50, color='blue', alpha=0.7)
# plt.title('Input Images Pixel Distribution')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.subplot(1, 2, 2)
# plt.hist(trans_images.flatten(), bins=50, color='green', alpha=0.7)
# plt.title('Trans Images Pixel Distribution')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()
# Example of creating a dataset with a filter
# dataset = OxAAADataset(data_root='/mnt/data/data/OxAAA/train/normalized', mode='train', filter='/mnt/data/data/OxAAA/train/qualifying_filename.npy')

