import os
import cv2

After_Dataset_dir = os.path.join(os.getcwd(),"After_Dataset")
After_Dataset_list = os.listdir(After_Dataset_dir)

Resized_Dataset_dir = os.path.join(os.getcwd(),"Resized_Dataset")


for dataset_num in After_Dataset_list:
    dataset_dir = os.path.join(After_Dataset_dir, dataset_num)
    dataset_list = os.listdir(dataset_dir)
    print(dataset_dir)
    for image_num in dataset_list:
        image_dir = os.path.join(dataset_dir, image_num)
        image = cv2.imread(image_dir)
        image_resize = cv2.resize(image,(256,256))
        resized_dataset_dir = os.path.join(Resized_Dataset_dir, dataset_num)
        cv2.imwrite(os.path.join(resized_dataset_dir, image_num),image_resize)

