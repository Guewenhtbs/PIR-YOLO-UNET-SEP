import SimpleITK as sitk
import os 

def filter(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        raw_img_sitk= sitk.ReadImage(input_path, sitk.sitkFloat32)
        raw_img_np = sitk.GetArrayFromImage(raw_img_sitk)
        raw_imp_np = raw_img_np[raw_img_np.shape[0]//3:raw_img_np.shape[0]-raw_img_np.shape[0]//6,:,:]
        raw_img_sitk = sitk.GetImageFromArray(raw_imp_np)
        filter = sitk.NormalizeImageFilter()
        filtered_img_sitk = filter.Execute(raw_img_sitk)
        output_path = os.path.join(output_dir, filename)
        sitk.WriteImage(filtered_img_sitk, output_path)
        print("Image filtered : ", filename)

filter("new_database/Skulled","new_database/Normed")