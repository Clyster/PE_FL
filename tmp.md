'''python
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from bts_dataloader import BtsModel
from PIL import Image
import matplotlib.pyplot as plt

def load_model(checkpoint_path, model_name, params):
    sys.path.append(os.path.dirname(checkpoint_path))
    model = BtsModel(params=params)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    return model

def process_image(image_path, do_kb_crop=True):
    image = cv2.imread(image_path)
    if do_kb_crop:
        height, width = image.shape[:2]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批量维度
    return image

def predict(model, image):
    with torch.no_grad():
        focal = torch.tensor([1.0]).cuda()  # 假设焦距为1.0, 可以根据实际情况调整
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image.cuda(), focal)
    return depth_est

def main(image_path, checkpoint_path, do_kb_crop=True):
    # 设置模型参数
    class Params:
        def __init__(self):
            self.model_name = 'bts_nyu_v2'
            self.encoder = 'densenet161_bts'
            self.data_path = ''
            self.filenames_file = ''
            self.input_height = 480
            self.input_width = 640
            self.max_depth = 80
            self.checkpoint_path = checkpoint_path
            self.dataset = 'nyu'
            self.do_kb_crop = do_kb_crop
            self.save_lpg = False
            self.bts_size = 512

    params = Params()
    model = load_model(checkpoint_path, params.model_name, params)
    image = process_image(image_path, do_kb_crop)
    depth_est = predict(model, image)
    
    # 可视化或保存结果
    depth_est = depth_est.squeeze().cpu().numpy()
    plt.imshow(depth_est, cmap='plasma')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'  # 输入图像路径
    checkpoint_path = 'path/to/your/checkpoint.pth'  # 模型检查点路径
    main(image_path, checkpoint_path, do_kb_crop=True)

'''python
