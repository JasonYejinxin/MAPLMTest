import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 BLIP 处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", ignore_mismatched_sizes=True)

model = model.to(device)

# 假设qa.json文件是存储在当前目录下
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTrain.json"  # 替换为你的qa.json文件路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

# 载入图像（每个frame文件夹只加载第一张图片）
def load_images_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train", frame_id)  # 指定图片路径

    # 检查文件夹是否存在
    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None  # 返回None，表示该frame没有对应的图片

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:1]:  # 只加载第一张图片
        img_path = os.path.join(frame_path, img_name)
        print(img_path)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None  # 如果没有图片，返回None

# 测试模型
def test_model(qa_data):
    model.eval()  # 设置模型为评估模式
    for data in qa_data:
        question = data["question"]
        frame = data["frame"]

        # 加载图像
        images = load_images_from_frame(frame)
        
        if images is None:
            continue
        
        # 提取图像特征
        pixel_values = processor(images=images[0], return_tensors="pt").pixel_values.to(device)
        
        # 将问题转化为输入 ID
        text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # 使用模型生成回答
        with torch.no_grad():  # 不需要计算梯度
            outputs = model.generate(input_ids=text_inputs, pixel_values=pixel_values)

        # 解码生成的答案
        generated_answer = processor.decode(outputs[0], skip_special_tokens=True)
        
        # 输出问题和模型生成的答案
        print(f"Question: {question}")
        print(f"Generated Answer: {generated_answer}")

# 启动模型测试
test_model(qa_data)
