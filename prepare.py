"""
Author  : Xu fuyong
Time    : created by 2019/7/17 19:32

"""
import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y

from tqdm import tqdm
import random
import os

images_dir = r"E:\sensingX\super-resolution\DRealSR\x2\Train_x2\train_HR"   # HR图片路径 
output_dir = r".\datasets"  # hdf5文件输出路径


total_num = 5
val_percent = 0.2                                      
scale = 2                                                       
seed = 4396

# 似乎是一些数据增强时用的参数
patch_size = 33
stride = 14


#训练集, 验证集图片数
val_num = int(total_num * val_percent)
train_num = total_num - val_num

# hdf5输出文件名(可自定义)
train_name = f"x{scale}_train_{train_num}"
val_name = f"x{scale}_val_{val_num}"


def train(img_list):
    print("---creating train set---")
    h5_file = h5py.File(os.path.join(output_dir, train_name), 'w')

    lr_patches = []
    hr_patches = []
    with tqdm(total = int(len(img_list))) as pbar:
        for image_path in img_list:
            hr = pil_image.open(image_path).convert('RGB')
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
            lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            for i in range(0, lr.shape[0] - patch_size + 1, stride):
                for j in range(0, lr.shape[1] - patch_size + 1, stride):
                    lr_patches.append(lr[i:i + patch_size, j:j + patch_size])
                    hr_patches.append(hr[i:i + patch_size, j:j + patch_size])
            pbar.update(1)

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    print("---saving file---")
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()
    print("Train set created")


def val(img_list):
    print("---creating val set---")
    h5_file = h5py.File(os.path.join(output_dir, val_name), 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    
    with tqdm(total = int(total_num * val_percent)) as pbar:
        for i, image_path in enumerate(img_list):
            hr = pil_image.open(image_path).convert('RGB')
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
            lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)
            pbar.update(1)

            lr_group.create_dataset(str(i), data=lr)
            hr_group.create_dataset(str(i), data=hr)
    h5_file.close()
    print("Val set created")

if __name__ == '__main__':
    total_list = glob.glob('{}/*'.format(images_dir))
    val_list = random.sample(total_list, k=int(total_num * val_percent))  # 从total_list列表中随机抽取k个
    new_total_list = [n for i, n in enumerate(total_list) if i not in val_list]
    train_list = random.sample(total_list, k=int(total_num * (1 - val_percent)))  # 从total_list列表中随机抽取k个
    
    print(f"Total Num:{len(total_list)}")
    print(f"Train set:{len(train_list)}")
    print(f"Val set:{len(val_list)}")

    train(train_list)
    val(val_list)
