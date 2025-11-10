import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from main import *

# 设置图片目录和模型路径
image_dir = "./tests/"
model_path = "best_model.pth"

# 数据转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载网络模型
model = VisionNet()
model.load_state_dict(torch.load(model_path))
model.eval() # 取消随机性

# 类别名称
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# 图片格式
image_formats = ["*.png", "*.jpg", "*.jpeg"]

# 获取所有匹配的图片文件路径
image_files = []
for fmt in image_formats:
    image_files.extend(glob.glob(os.path.join(image_dir, fmt)))

# 遍历每张图片
for image_path in image_files:
    image = Image.open(image_path)
    image = image.convert('RGB')
    
    # 应用转换
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度
    
    # 模型预测
    with torch.no_grad():
        output = model(image)
    
    # 结果
    predicted_class = output.argmax(1).item()
    predicted_class_name = class_names[predicted_class]
    print(f"Image: {image_path}, Predicted class: {predicted_class_name}")