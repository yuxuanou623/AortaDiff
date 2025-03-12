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
from pathlib import Path

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


def get_brats2021_train_transform_abnormalty_train(image_size):
    base_transform = get_brats2021_base_transform_abnormalty_train(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'trans', 'brainmask', 'seg']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_brats2021_base_transform_abnormalty_train(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'trans', 'brainmask', 'seg']),
        transforms.Resized(
            keys=['input', 'trans', 'brainmask', 'seg'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform

def get_oxaaa_base_transform_abnormalty_test(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'trans']),
        transforms.Resized(
            keys=['input', 'trans'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform
def get_oxaaa_base_transform_abnormalty_train(image_size):
    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'trans']),
        transforms.Resized(
            keys=['input', 'trans'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform




def get_oxaaa_train_transform_abnormalty_test(image_size):
    base_transform = get_oxaaa_base_transform_abnormalty_test(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'trans']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_oxaaa_train_transform_abnormalty_train(image_size):
    base_transform = get_oxaaa_base_transform_abnormalty_train(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'trans']),
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
    def __init__(self, data_root: str, mode: str, input_mod='noncontrast', trans_mod='contrast', transforms=None, filter=None):
        super(OxAAADataset, self).__init__()
        assert mode in ['train', 'test', 'val'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.input_mod = input_mod  # Typically 'noncon'
        self.trans_mod = trans_mod  # Typically 'con'
        self.transforms = transforms
        
        self.data_root =  Path(self.data_root)
       
        # Initialize directories for contrast and non-contrast images
        self.input_dir = Path(self.data_root) / self.input_mod
        self.trans_dir = Path(self.data_root) / self.trans_mod

        # Load the filter if provided
        if filter is not None:
            self.filter = np.load(filter, allow_pickle=True)  # Load filenames from the npy file
            self.filter = set(self.filter.tolist())  # Convert to set for faster lookup
        else:
            self.filter = None
        
        # List all image names in the input directory, possibly filtered
        self.input_images = sorted([img for img in self.input_dir.glob('*.nii.gz') if self.filter is None or img.name.split('/')[0] in self.filter])
       

        # Dictionary to quickly find corresponding images
        self.image_pairs = self._cache_pairs()
        

    def _cache_pairs(self):
        pairs = {}
        # Pair images with the same name in input and trans directories
        for input_img in self.input_images:
            trans_img_path = self.trans_dir / input_img.name
            if trans_img_path.exists():
                pairs[input_img] = trans_img_path
        return pairs

    def __getitem__(self, index):
        input_img_path, trans_img_path = list(self.image_pairs.items())[index]
        # Load images
        input_image = nib.load(input_img_path).get_fdata()
        trans_image = nib.load(trans_img_path).get_fdata()

        data_dict = {'input': input_image, 'trans': trans_image}
        # Transform data if transforms are provided
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
# batch_data = [dataset[i] for i in range(2)]

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

