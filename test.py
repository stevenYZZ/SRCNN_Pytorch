"""
Author  : Xu fuyong
Time    : created by 2019/7/17 17:41

"""
import os
import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from model import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, combine_test_image


input_dir = "./data"
# image_name = "butterfly_GT.bmp"
# image_name = "ppt3.bmp"
image_name = "zebra.bmp"
model_file = "./outputs/x3/best.pth"

input_file = os.path.join(input_dir, image_name)
# output_dir = os.path.join(input_dir, "combine")
scale = 3


if __name__ == '__main__':
    cudnn.benchmark = True
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(model_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(input_file).convert('RGB')

    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

    image.save(os.path.splitext(input_file)[0] + '_bicubic_x{}.bmp'.format(scale))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(os.path.splitext(input_file)[0] + '_srcnn_x{}.bmp'.format(scale))
    
    # combine图像从左到右依次为输入图(LR)、输出图(predict)、真值图(HR)
    combine_test_image(input_dir, input_dir, image_name, scale)

