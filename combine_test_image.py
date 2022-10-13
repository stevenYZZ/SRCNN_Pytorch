import os
from PIL import Image
import numpy as np

input_path = "./try"
output_path = "./try/combine"

# file_name = "butterfly_GT.bmp"
# file_name = "ppt3.bmp"
file_name = "zebra.bmp"

scale = 3

if(not os.path.exists(output_path)):
    os.makedirs(output_path)

img_ground_truth = Image.open(os.path.join(input_path, file_name))
img_input = Image.open(os.path.join(input_path, os.path.splitext(file_name)[0] + '_bicubic_x{}.bmp'.format(scale)))
img_output = Image.open(os.path.join(input_path, os.path.splitext(file_name)[0] + '_srcnn_x{}.bmp'.format(scale)))

cbox = [0,0,img_input.size[0],img_input.size[1]]
img_ground_truth = img_ground_truth.crop(cbox)
# print(img_ground_truth.size)
basemat=np.atleast_2d(img_input)
for im in [img_output, img_ground_truth]:
    mat=np.atleast_2d(im)
    
    
    # print(mat.shape)
    # exit()
    # print(mat.shape)
    basemat = np.concatenate((basemat, mat), axis = 1)
    # print(basemat.shape)
    
# print(basemat)
# 图片从数组转换回来
final_img=Image.fromarray(basemat)
final_img.save(os.path.join(output_path, file_name))
# print

# print(img_ground_truth.size)

# image_combine = Image.new('RGB', image1.size)