# 출처 : https://yeeca.tistory.com/23

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.target_activations = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.target_activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]
        
        target_layer = self.get_target_layer()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def get_target_layer(self):
        if isinstance(self.target_layer, str):
            modules = dict([*self.model.named_modules()])
            print("layer : str")
            return modules[self.target_layer]
        elif isinstance(self.target_layer, int):
            children = [*self.model.children()]
            print("layer : index")
            return children[self.target_layer]
        elif isinstance(self.target_layer, nn.Sequential):
            print("layer : nn.sequential")
            return self.target_layer[-1]
        else:
            raise ValueError("Invalid target layer value. It should be either the layer name (str), index (int), or nn.Sequential.")
    
    def forward_pass(self, input_tensor):
        return self.model(input_tensor)
    
    def backward_pass(self, output):
        one_hot = torch.zeros_like(output, dtype=torch.float)
        one_hot[0][torch.argmax(output)] = 1.0

        output.backward(gradient=one_hot)
        print(f"debug : class : {[torch.argmax(output)]}") # https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/ 에서 torch.argmax(output)에 해당하는 클래스명을 찾을 수 있음
    
    def generate_gradcam(self, input_tensor):
        self.model.zero_grad()
        output = self.forward_pass(input_tensor)
        self.backward_pass(output)

        gradients = self.gradient.cpu().numpy()
        target_activations = self.target_activations.cpu().numpy()

        weights = np.mean(gradients, axis=(2,3), keepdims=True)
        gradcam = np.sum(weights * target_activations, axis=1)
        gradcam = np.maximum(gradcam, 0)

        # gradcam = gradcam[:,:,0]
        print(gradcam.shape)
        gradcam = np.squeeze(gradcam, 0)
        gradcam = cv2.resize(gradcam, (224, 224))
        gradcam = (gradcam - np.min(gradcam)) / (np.max(gradcam) - np.min(gradcam) + 1e-7)

        return gradcam

def visualize_gradcam(image_path, gradcam):
    image = np.array(Image.open(image_path))
    image = cv2.resize(image, (224,224))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(gradcam, cmap="jet", alpha=0.7)
    plt.imshow(image, alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")

    # plt.show()
    plt.savefig("test.jpg")
    print(image.shape, gradcam.shape)

if __name__ == "__main__":
    from torchvision.models import resnet50

    def load_model():
        model = resnet50(pretrained=True)
        model.eval()
        return model
    
    def get_transform():
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform
    
    def process_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = get_transform()(image).unsqueeze(0)
        return image_tensor
    
    def visualize_gradcam_test(image_path, gradcam):
        image = np.array(Image.open(image_path))
        image = cv2.resize(image, (224, 224))

        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("Original image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(gradcam, cmap="jet", alpha=0.7)
        plt.imshow(image, alpha=0.5)
        plt.title("Grad-CAM")
        plt.axis("off")

        #plt.show()
        plt.savefig("test.jpg")
        print(image.shape, gradcam.shape)

    model = load_model()

    for name, module in model.named_children():
        print(name)

    pipeline = GradCam(model, 'layer4')
    image_path = f"./images/PetImages/Cat/1.jpg" # 이미지 경로 지정
    
    image_tensor = process_image(image_path)
    gradcam = pipeline.generate_gradcam(image_tensor)

    visualize_gradcam(image_path, gradcam)
