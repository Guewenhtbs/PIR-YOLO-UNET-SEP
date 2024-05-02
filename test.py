import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf

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


