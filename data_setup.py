import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
#import tensorflow as tf
from pathlib import Path
import cv2 as cv2
from PIL import Image

# 48/53/54/56/57/14/23/32/41
bad_p = []
flair_ref = sitk.ReadImage("Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information/Patient-9/9-Flair.nii", sitk.sitkFloat32)


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
    sitk_flair = sitk.ReadImage(volume_file, sitk.sitkFloat32)
    sitk_flair_seg = sitk.ReadImage(seg_file)


    # Resample the image to the same size as the reference image:
    
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(64)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.2)
    r.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=500, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    r.SetInterpolator(sitk.sitkBSpline)
    r.SetOptimizerScalesFromPhysicalShift()

    tx = sitk.CenteredTransformInitializer(flair_ref, sitk_flair, sitk.AffineTransform(sitk_flair.GetDimension()), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    r.SetInitialTransform(tx)
    
    outTx = r.Execute(flair_ref, sitk_flair)
    sitk_flair_resample = sitk.Resample(sitk_flair, flair_ref, outTx, sitk.sitkBSpline, 0.0, sitk_flair.GetPixelIDValue())
    sitk_flair_seg_resample = sitk.Resample(sitk_flair_seg, flair_ref, outTx, sitk.sitkNearestNeighbor, 0.0, sitk_flair_seg.GetPixelIDValue())
    print(outTx)
    print(patient_number,sitk_flair.GetSize(),sitk_flair_resample.GetSize())
    
    

    # and access the numpy array :
    ar_flair = sitk.GetArrayFromImage(sitk_flair_resample)
    ar_flair_seg = sitk.GetArrayFromImage(sitk_flair_seg_resample)

    # Slice on the axial plane all images who have a lesion:
    if patient_number not in bad_p:
        for i in range(len(ar_flair_seg)) :
            image = Image.fromarray((((ar_flair[i,:,:]-ar_flair[:,:,:].min())/(ar_flair[:,:,:].max()-ar_flair[:,:,:].min()))* 255).astype(np.uint8))
            image.save(img_path / f"img_{patient_number}{i:04}.png")
            """ if ar_flair_seg[i,:,:].max() > 0 :
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
                    image = Image.fromarray(((ar_flair[i,:,:]/ar_flair[:,:,:].max())* 255).astype(np.uint8))
                    image.save(img_path / f"img_{patient_number}{i}.png")
 """

def GetPatientPath(n):
    raw_data_path = Path("Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information")
    
    return raw_data_path / f"Patient-{n}" / f"{n}-Flair.nii" , raw_data_path / f"Patient-{n}" / f"{n}-LesionSeg-Flair.nii"


def GenDataYOLO(data_root_path,train_num,val_num,test_num) :
    data_root_path = Path(data_root_path)

    for i in range(1, train_num+1):
        flair, seg = GetPatientPath(i)
        ReadVolumestoYolo(flair,seg, data_root_path / "train",i) 
        
    for i in range(train_num+1, val_num + train_num+1):
        flair, seg = GetPatientPath(i)
        ReadVolumestoYolo(flair,seg, data_root_path / "val",i) 
        
    for i in range(val_num + train_num+1, test_num + val_num + train_num+1):
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


