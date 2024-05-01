import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# A path to a T1-weighted brain .nii image:
flair = r'c:\Users\kergu.LAPTOP-RGB94A60\Documents\TC\PIR\Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information\Patient-1\1-Flair.nii'

flair_seg = r'c:\Users\kergu.LAPTOP-RGB94A60\Documents\TC\PIR\Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information\Patient-1\1-LesionSeg-Flair.nii'

# Read the .nii image containing the volume with SimpleITK:
sitk_flair = sitk.ReadImage(flair)
sitk_flair_seg = sitk.ReadImage(flair_seg)

# and access the numpy array:
ar_flair = sitk.GetArrayFromImage(sitk_flair)
ar_flair_seg = sitk.GetArrayFromImage(sitk_flair_seg)

normalizer = sitk.NormalizeImageFilter()

n_flair = normalizer.Execute(sitk_flair)

ar_n_flair = sitk.GetArrayFromImage(n_flair)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ar_flair[9,:,:], cmap='gray')
axes[0].set_title('Original Flair')

axes[1].imshow(ar_n_flair[9,:,:], cmap='gray')
axes[1].set_title('Normalized Flair')

axes[2].imshow(ar_flair_seg[9,:,:], cmap='gray')
axes[2].set_title('Flair Segmentation')

plt.show()

for i in range(23) :
    if ar_flair_seg[i,:,:].max() > 0 :
        print(i)