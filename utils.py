from PIL import Image
import torchvision
import torch
import sys
import os
import warnings

root_path = "/home/qiufengwang/code/xingyc/face.evoLVe.PyTorch/"
data_root = os.path.join(root_path, 'data/our_face')
crop_root = os.path.join(root_path, 'data/crop_face')
gen_root = os.path.join(root_path, 'data/gen_face')

sys.path.append(os.path.join(root_path, 'align'))
sys.path.append(os.path.join(root_path, 'backbone'))

warnings.filterwarnings("ignore")

to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                  ])
to_pil_image = torchvision.transforms.Compose([torchvision.transforms.Normalize([-1, -1, -1], [2, 2, 2]),
                                               torchvision.transforms.ToPILImage()
                                               ])


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def get_adv_sample(noise, img_origin):
    noise = noise.clamp_(-0.2, 0.2)
    img_add_noise = to_pil_image((noise + img_origin).clamp_(-1, 1)[0].cpu())
    return img_add_noise


# 计算noise的二范数，也是天池人脸识别进攻使用的评价标准
def get_statistics(noise, img_origin):
    noise = (noise + img_origin).clamp_(-1, 1) - img_origin
    l2 = torch.mean(torch.sqrt(torch.sum((noise * 127.5) ** 2, 0)))
    return l2


def list2map(li):
    m = {}
    for i in range(len(li)):
        m[li[i]] = i
    return m
