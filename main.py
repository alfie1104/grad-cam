# 출처) https://yonghip.tistory.com/entry/Pytorch-GradCAM-%EC%B9%98%ED%8A%B8%EC%8B%9C%ED%8A%B8

import timm # torchvision에서 제공하는 pretrained model보다 더 많은 사전학습된 모델을 제공함 (pytorch에서 활용가능한 사전학습 모델 제공. huggingface에서 다운받음음)

import os
import random
from collections import defaultdict
from importlib import import_module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import random
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.datasets import ImageFolder

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# 데이터셋 경로 지정
IMAGE_PATH = "./images/PetImages"

def make_dataset():
    dataset = ImageFolder(IMAGE_PATH)

    # dont need to make data_loader if you just want to visualize random image in dataset
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=3, # args.batch_size
        shuffle=True,
        num_workers=0, # cpu 코어 절반
        drop_last=False
    )

    return dataset, train_loader

# 데이터 셋과 모델 지정
dataset, train_loader = make_dataset()

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)

target_layers = [model.conv_head]

fig, axes = plt.subplots(4, 4, figsize=(14,10))
axes = axes.flatten()

for i in range(16):
    img = torch.tensor(np.array(dataset[i][0])).float()
    img /= 255.0
    label = torch.tensor(dataset[i][1])
    targets = [BinaryClassifierOutputTarget(label)]
    img = img.permute(2,0,1)
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img.unsqueeze(0).float(), targets=targets)
    img = img.permute(1,2,0)

    visualization = show_cam_on_image(np.float32(img), grayscale_cam[0, :], use_rgb=True)
    axes[i].imshow(visualization)
    axes[i].axis("off")

plt.show()
"""
아래 코드를 통해 model의 layer이름을 알 수 있음
for name, ch in model.named_children():
    print("name :", name)
    print("child :", ch)
    print("==========================")
이 중 CNN 모듈의 이름을 다음과 같이 target_layer에 넣어 원하는 층이 이미지의 특정 부분을 얼마나 집중했는지 볼 수 있음
target_layers = [model.conv_head]
"""


