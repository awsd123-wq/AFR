import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from torch.autograd import Function

# 加载模型 (这里以 ResNet50 为例, 你需要替换为自己的模型)
model = resnet50(pretrained=True)
model.eval()

# 预处理视频关键帧
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 读取关键帧
image_path = r"E:\深度学习实验复现\CMSIL-main\red.png"  # 关键帧图片路径
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = transform(image).unsqueeze(0)

# 选取目标层（如 ResNet 的最后一层卷积层）
target_layer = model.layer4[-1]


# 计算 Grad-CAM
def get_gradcam_heatmap(model, target_layer, input_tensor):
    gradients = []
    activations = []

    def hook_fn(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook_fn(module, input, output):
        activations.append(output)

    # 注册钩子
    handle = target_layer.register_forward_hook(forward_hook_fn)
    handle_grad = target_layer.register_backward_hook(hook_fn)

    # 前向传播
    output = model(input_tensor)
    score = output[:, output.argmax(dim=1)]

    # 反向传播
    model.zero_grad()
    score.backward(retain_graph=True)

    # 获取梯度和激活值
    grad = gradients[0].cpu().data.numpy()
    activation = activations[0].cpu().data.numpy()

    # 计算 Grad-CAM 热力图
    weights = np.mean(grad, axis=(2, 3), keepdims=True)
    cam = np.sum(weights * activation, axis=1)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    return cam[0]


# 计算热力图
heatmap = get_gradcam_heatmap(model, target_layer, input_tensor)

# 叠加热力图到原始图片
heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 叠加到原始帧
superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# 显示图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Frame")

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title("Grad-CAM Heatmap")

plt.show()
