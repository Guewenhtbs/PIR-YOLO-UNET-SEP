import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import cv2 as cv2
from PIL import Image


def ReadVolumestoYolo(volume_file,seg_file,save_path,patient_number) :
    """
    Read the volume and the segmentation of a MRI image and write files in the format for the YOLO model.
    Parameters:
    volume_file: str
        Path to the volume file.
    seg_file: str
        Path to the segmentation file.
    save_path: str
        Path to save the images and the segmentations in the YOLO format.
    """
    img_path = save_path / "images"
    labels_path = save_path / "labels"

    img_path.mkdir( parents=True, exist_ok=True )
    labels_path.mkdir( parents=True, exist_ok=True )

    # Read the .nii image containing the volume with SimpleITK:
    sitk_flair = sitk.ReadImage(volume_file)
    sitk_flair_seg = sitk.ReadImage(seg_file)

    # and access the numpy array :
    ar_flair = sitk.GetArrayFromImage(sitk_flair)
    ar_flair_seg = sitk.GetArrayFromImage(sitk_flair_seg)

    # Slice on the axial plane all images who have a lesion:
    for i in range(len(ar_flair_seg)) :
        if ar_flair_seg[i,:,:].max() > 0 :
            segmented_values = False
            
            # Write the segmentation in the YOLO format:
            mask = (ar_flair_seg[i,:,:]> 0).astype('uint8') * 255
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            with open(labels_path / f"img_{patient_number}{i}.txt", "w") as f:
                for j in range(len(contours)) :
                    if contours[j].shape[0] > 2 :
                        segmented_values = True
                        contour = contours[j].flatten().tolist()
                        f.write(f"{0}")
                        for k in range(len(contour)) :
                            f.write(f" {contour[k]/256}")
                        f.write("\n")

            # Write the image in the YOLO format
            if segmented_values :
                image = Image.fromarray(((ar_flair[i,:,:]/ar_flair[i,:,:].max())* 255).astype(np.uint8))
                image.save(img_path / f"img_{patient_number}{i}.png")


def GetPatientPath(n):
    raw_data_path = Path("raw_data/Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information")
    
    return raw_data_path / f"Patient-{n}" / f"{n}-Flair.nii" , raw_data_path / f"Patient-{n}" / f"{n}-LesionSeg-Flair.nii"


def GenDataYOLO(data_root_path,train_num,val_num,test_num) :
    data_root_path = Path(data_root_path)

    for i in range(1, train_num):
        flair, seg = GetPatientPath(i)
        ReadVolumestoYolo(flair,seg, data_root_path / "train",i) 
        
    for i in range(train_num, val_num + train_num):
        flair, seg = GetPatientPath(i)
        ReadVolumestoYolo(flair,seg, data_root_path / "val",i) 
        
    for i in range(val_num + train_num, test_num + val_num + train_num):
        flair, seg = GetPatientPath(i)
        ReadVolumestoYolo(flair,seg, data_root_path / "test",i) 


GenDataYOLO("datasets",train_num=38,val_num=10,test_num=12)
    
yaml_content = f"""
train: train/images
val: val/images
test: test/images

names: ['lesions']
    """
    
with Path('data.yaml').open('w') as f:
    f.write(yaml_content)


