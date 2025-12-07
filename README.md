# 🫀 AortaDiff / AortaDiff-P
### *Diffusion and UNet-based Aortic Reconstruction and Lumen Masking Framework*

![Visualization](visualization/figure.jpg)

AortaDiff is a multitasking diffusion model that can jointly do image translation from NCCT to CECT and segment lumen and thrombus mask. Unlike previous multitask diffusion models, our approach requires no initial predictions (e.g., a coarse segmentation mask), shares both encoder and decoder parameters across tasks, and employs a semi-supervised training strategy to learn from scans with missing segmentation labels, a common constraint in clinical data.
Evaluated on a cohort of 264 patients, our method consistently outperformed state-of-the-art single-task and multi-stage models. For image synthesis, it achieved a PSNR of 25.61 dB, compared to 23.80 dB from a single-task CDM. For segmentation, it improved the lumen Dice score to 0.89 from 0.87 and the challenging thrombus Dice score to 0.53 from 0.48 (nnU-Net). These segmentation enhancements led to more accurate clinical measurements, reducing the lumen diameter MAE to 4.19 mm from 5.78 mm and the thrombus area error to 33.85\% from 41.45\%. 



## 📦 Usage

### 1️⃣ Data Preparation

Prepare the dataset following these requirements:

#### ✔ Normalize images
Normalize **noncontrast** and **contrast** CT images to the range **[-1, 1]**.

#### ✔ Consistent pairing
If noncontrast and contrast belong to the same case,  
they must share the **same filename**.

#### ✔ Required masks
Prepare the following masks:

| Filename | Description |
|----------|-------------|
| `contrast` | Contrast CT image |
| `noncontrast` | Noncontrast CT image |
| `noncontrastmask` | Aorta mask for noncontrast CT image |
| `contrastaortamask` | Aorta mask for contrast CT image |
| `contrastlumenmask` | Lumen mask for contrast CT image |

Place *all images and masks in the **same directory***.



