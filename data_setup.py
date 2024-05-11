import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
#import tensorflow as tf
from pathlib import Path
import cv2 as cv2
from PIL import Image
import os

# 48/53/54/56/57/14/23/32/41
# bad_p = [48,53,54,56,57,14,23,32,41]
# flair_ref = sitk.ReadImage("new_database/Flair/9-Flair.nii", sitk.sitkFloat32)
# flair_ref.SetSpacing([1,1,1])
# r = sitk.ImageRegistrationMethod()
# r.SetMetricAsMattesMutualInformation(64)
# r.SetMetricSamplingStrategy(r.RANDOM)
# r.SetMetricSamplingPercentage(0.1)
# r.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=500, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
# r.SetInterpolator(sitk.sitkBSpline)
# r.SetOptimizerScalesFromPhysicalShift()

openingFilter = sitk.BinaryMorphologicalOpeningImageFilter()
openingFilter.SetKernelRadius(4)
openingFilter.SetKernelType(sitk.sitkBall)


corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetMaximumNumberOfIterations([20]*4)

caster = sitk.CastImageFilter()
caster.SetOutputPixelType(sitk.sitkUInt8)

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
    # if patient_number not in bad_p:
    img_path = save_path / "images"
    labels_path = save_path / "labels"
    seg_path = save_path / "segmentations"

    img_path.mkdir( parents=True, exist_ok=True )
    labels_path.mkdir( parents=True, exist_ok=True )
    seg_path.mkdir( parents=True, exist_ok=True )

    # Read the .nii image containing the volume with SimpleITK:
    sitk_flair = sitk.ReadImage(volume_file, sitk.sitkFloat32)
    sitk_flair_seg = sitk.ReadImage(seg_file)
    # coef = min(flair_ref.GetSize()[0]/sitk_flair.GetSize()[0],flair_ref.GetSize()[1]/sitk_flair.GetSize()[1])
    # scale = [coef,coef,1]
    # sitk_flair.SetSpacing(scale)
    # sitk_flair_seg.SetSpacing(scale)

    # # Resample the image to the same size as the reference image:
    # tx = sitk.CenteredTransformInitializer(flair_ref, sitk_flair, sitk.AffineTransform(sitk_flair.GetDimension()), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # r.SetInitialTransform(tx)
    
    # outTx = r.Execute(flair_ref, sitk_flair)
    # sitk_flair_resample = sitk.Resample(sitk_flair, flair_ref, outTx, sitk.sitkBSpline, 0.0, sitk_flair.GetPixelIDValue())
    # sitk_flair_seg_resample = sitk.Resample(sitk_flair_seg, flair_ref, outTx, sitk.sitkNearestNeighbor, 0.0, sitk_flair_seg.GetPixelIDValue())
    # print(outTx)
    

    # N4 filtering
    maskImage = sitk.OtsuThreshold(sitk_flair, 0, 1,200)
    print(f"N4 filtering {patient_number} ...")
    openedMaskImage = openingFilter.Execute(maskImage)
    corrected_image = corrector.Execute(sitk_flair, openedMaskImage)
    
    # and access the numpy array :
    # ar_flair = sitk.GetArrayFromImage(corrected_image)
    ar_flair_seg = sitk.GetArrayFromImage(sitk_flair_seg)
    print(patient_number,sitk_flair.GetSize(),sitk_flair_seg.GetSize())
    

    # Slice on the axial plane all images who have a lesion:

    for i in range(len(ar_flair_seg)) :   
        if ar_flair_seg[i,:,:].max() > 0 :
            segmented_values = False
            
            # Write the segmentation in the YOLO format:
            mask = (ar_flair_seg[i,:,:]> 0).astype('uint8') * 255
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            with open(labels_path / f"img_{patient_number:02}{i:04}.txt", "w") as f:
                for j in range(len(contours)) :
                    if contours[j].shape[0] > 2 :
                        segmented_values = True
                        contour = contours[j].flatten().tolist()
                        f.write(f"{0}")
                        for k in range(len(contour)) :
                            if k % 2 == 0 :
                                f.write(f" {contour[k]/ar_flair_seg.shape[2]}")
                            else :
                                f.write(f" {contour[k]/ar_flair_seg.shape[1]}")
                        f.write("\n")

            # Write the image in the YOLO format
            if segmented_values :
                # ar_image = (((ar_flair[i,:,:]-ar_flair[:,:,:].min())/(ar_flair[:,:,:].max()-ar_flair[:,:,:].min()))* 255).astype(np.uint8)
                # image = Image.fromarray(ar_image)
                # image.save(img_path / f"img_{patient_number:02}{i:04}.png")
                seg_np = Image.fromarray(mask)
                seg_np.save(seg_path / f"np_{patient_number:02}{i:04}.png")
                
                image = sitk.RescaleIntensity(corrected_image[:,:,i],0,255)
                image = caster.Execute(image)
                sitk.WriteImage(image, str(img_path / f"img_{patient_number:02}{i:04}.png"))

                # # ne fonctionne pas jsp pq
                # seg = caster.Execute(sitk_flair_seg[:,:,i])
                # sitk.WriteImage(seg, str(seg_path / f"img_{patient_number:02}{i:04}.png"))

                # plt.imshow(ar_image, cmap='gray')
                # plt.imshow(mask, cmap='jet', alpha=0.5)
                # plt.show()
                # im = cv2.imread(str(img_path / f"img_{patient_number:02}{i:04}.png"))   
                # cv2.drawContours(im, contours, -1, (0,255,0), 1)
                # cv2.imshow('None approximation', im)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                


# def GetPatientPath(n):
#     raw_data_path = Path("new_database")
    
#     return raw_data_path / "Flair" / f"{n}-Flair.nii" , raw_data_path / "LesionSeg" / f"{n}-LesionSeg-Flair.nii"


def GenDataYOLO(data_root_path,train_num,val_num,test_num) :
    data_root_path = Path(data_root_path)
    data_root_path.mkdir(parents=True, exist_ok=True)
    
    inp_path = Path("0_Data_reg_inter_rigid")
    filenames = os.listdir(inp_path)
    for i in range(1, train_num+1):
        flair = inp_path / filenames[i-1] / "3DFLAIR.nii"
        seg = inp_path / filenames[i-1] / "Consensus.nii"
        ReadVolumestoYolo(flair,seg, data_root_path / "train",i)

    for i in range(train_num+1, val_num + train_num+1):
        flair = inp_path / filenames[i-1] / "3DFLAIR.nii"
        seg = inp_path / filenames[i-1] / "Consensus.nii"
        ReadVolumestoYolo(flair,seg, data_root_path / "val",i)

    for i in range(val_num + train_num+1, test_num + val_num + train_num+1):
        flair = inp_path / filenames[i-1] / "3DFLAIR.nii"
        seg = inp_path / filenames[i-1] / "Consensus.nii"
        ReadVolumestoYolo(flair,seg, data_root_path / "test",i)
    
    # for i in range(1, train_num+1):
    #     flair, seg = GetPatientPath(i)
    #     ReadVolumestoYolo(flair,seg, data_root_path / "train",i) 
        
    # for i in range(train_num+1, val_num + train_num+1):
    #     flair, seg = GetPatientPath(i)
    #     ReadVolumestoYolo(flair,seg, data_root_path / "val",i) 
        
    # for i in range(val_num + train_num+1, test_num + val_num + train_num+1):
    #     flair, seg = GetPatientPath(i)
    #     ReadVolumestoYolo(flair,seg, data_root_path / "test",i) 


GenDataYOLO("new_Miccai",train_num=9,val_num=3,test_num=3)
    
yaml_content = f"""
train: train/images
val: val/images
test: test/images

names: ['lesions']
    """
    
with Path('data.yaml').open('w') as f:
    f.write(yaml_content)


