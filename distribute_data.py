import os
import shutil

def create_datasets(src_root, dst_root, train_patients, val_patients, test_patients):

    os.makedirs(dst_root, exist_ok=True)
    
    for folder in ['train', 'val', 'test']:
        src_folder = os.path.join(src_root, folder)
        dst_folder = os.path.join(dst_root, folder)
        os.makedirs(dst_folder, exist_ok=True)
        

        src_images_folder = os.path.join(src_folder, 'images')
        dst_images_folder = os.path.join(dst_folder, 'images')
        src_labels_folder = os.path.join(src_folder, 'labels')
        dst_labels_folder = os.path.join(dst_folder, 'labels')
        os.makedirs(dst_images_folder, exist_ok=True)
        os.makedirs(dst_labels_folder, exist_ok=True)
            
        if folder == 'train':
            patients = train_patients
        elif folder == 'val':
            patients = val_patients
        elif folder == 'test':
            patients = test_patients
                

        for src_type_folder_name in os.listdir(src_root):
            src_type_folder = os.path.join(src_root, src_type_folder_name)
            
            src_images_folder = os.path.join(src_type_folder, 'images')
            for filename in os.listdir(src_images_folder):
                if filename.startswith("img_") and filename.endswith(".png"):

                    patient = int(filename.split("_")[1][:2])
                    
                    if patient in patients:
                        src_image_path = os.path.join(src_images_folder, filename)
                            
                        dst_image_path = os.path.join(dst_images_folder, filename)
                        
                        shutil.copy(src_image_path, dst_image_path)

            src_labels_folder = os.path.join(src_type_folder, 'labels')
            for filename in os.listdir(src_labels_folder):
                if filename.startswith("img_") and filename.endswith(".txt"):                
                    patient = int(filename.split("_")[1][:2])
                        
                    if patient in patients:
                            
                        src_label_path = os.path.join(src_labels_folder, filename)
                        dst_label_path = os.path.join(dst_labels_folder, filename)
  
                        shutil.copy(src_label_path, dst_label_path)



src_root = "datasets"
dst_roots = ["datasets1", "datasets2", "datasets3", "datasets4"]
train_patients_lists = [
    list(range(1, 7)) + list(range(13, 16)),
    list(range(1, 4)) + list(range(10, 16)),
    list(range(7, 16)),
    list(range(4, 13))
]
val_patients_lists = [
    list(range(7, 10)),
    list(range(4, 7)),
    list(range(1, 4)),
    list(range(13, 16))
]
test_patients_lists = [
    list(range(10, 13)),
    list(range(7, 10)),
    list(range(4, 7)),
    list(range(1, 4))
]

for i, (dst_root, train_patients, val_patients, test_patients) in enumerate(zip(dst_roots, train_patients_lists, val_patients_lists, test_patients_lists)):
    create_datasets(src_root, dst_root, train_patients, val_patients, test_patients)

