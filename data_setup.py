import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv2
from PIL import Image

def ReadVolumestoUnet(volume_file,seg_file) :
    """
    Read the volume and the segmentation of a MRI image and return the images in the format for the U-Net model.
    Parameters:
    volume_file: str
        Path to the volume file.
    seg_file: str
        Path to the segmentation file.
    Returns:
    X: np.array
        Array with the axial images.
    y: np.array
        Array with the segmentation images.
    """
    # Read the .nii image containing the volume with SimpleITK:
    sitk_flair = sitk.ReadImage(volume_file)
    sitk_flair_seg = sitk.ReadImage(seg_file)

    # and access the numpy array with a normalizer filter:
    normalizer = sitk.NormalizeImageFilter()
    n_flair = normalizer.Execute(sitk_flair)
    ar_n_flair = sitk.GetArrayFromImage(n_flair)

    ar_flair_seg = sitk.GetArrayFromImage(sitk_flair_seg)

    # Slice on the axial plane all images who have a lesion:
    X=[]
    y=[]
    for i in range(len(ar_flair_seg)) :
        if ar_flair_seg[i,:,:].max() > 0 :
            X.append(ar_n_flair[i,:,:])
            y.append(ar_flair_seg[i,:,:])
    X = np.asarray(X, dtype=np.float32)
    X = np.expand_dims(X,-1)
    y = np.asarray(y, dtype=np.float32)
    y = tf.keras.utils.to_categorical(y)

    return X,y

def ReadVolumestoYolo(volume_file,seg_file,save_path) :
    """
    Read the volume and the segmentation of a MRI image and write file in the format for the YOLO model.
    Parameters:
    volume_file: str
        Path to the volume file.
    seg_file: str
        Path to the segmentation file.
    """
    patient_number = volume_file.split('Patient-')[1].split('\\')[0]
    # Read the .nii image containing the volume with SimpleITK:
    sitk_flair = sitk.ReadImage(volume_file)
    sitk_flair_seg = sitk.ReadImage(seg_file)

    # and access the numpy array :
    ar_flair = sitk.GetArrayFromImage(sitk_flair)
    ar_flair_seg = sitk.GetArrayFromImage(sitk_flair_seg)

    # Slice on the axial plane all images who have a lesion:
    for i in range(len(ar_flair_seg)) :
        if ar_flair_seg[i,:,:].max() > 0 :
            # Write the image in the YOLO format
            image = Image.fromarray(((ar_flair[i,:,:]/ar_flair[i,:,:].max())* 255).astype(np.uint8))
            image.save(f"{save_path}Patient-{patient_number}_{i}.jpg")
            # Write the segmentation in the YOLO format:
            mask = (ar_flair_seg[i,:,:]> 0).astype('uint8') * 255
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            with open(f"{save_path}Patient-{patient_number}-LesionSeg_{i}.txt", "w") as f:
                for i in range(len(contours)) :
                    contour = contours[i].flatten().tolist()
                    f.write(f"{i}")
                    for j in range(len(contour)) :
                        f.write(f" {contour[j]}")


ReadVolumestoYolo(r'C:\Users\kergu.LAPTOP-RGB94A60\Documents\TC\PIR\Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information\Patient-1\1-Flair.nii',r'C:\Users\kergu.LAPTOP-RGB94A60\Documents\TC\PIR\Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information\Patient-1\1-LesionSeg-Flair.nii',r'C:\Users\kergu.LAPTOP-RGB94A60\Documents\TC\PIR\\')