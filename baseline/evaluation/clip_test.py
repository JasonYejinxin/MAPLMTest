import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json

# 设置设备为cuda或者cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载保存的模型和处理器
model_path = "./blip2_flan_t5_epoch_5"  # 保存模型的路径
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

# 将模型移至指定设备（GPU或CPU）
model = model.to(device)

# 测试图片路径
test_image_path = "/path/to/test_image.jpg"  # 替换为你的测试图像路径

# 测试问题
test_question = "What is the color of the sky?"  # 替换为你想测试的问题

# 载入图像
def load_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# 处理图像和文本数据
def test_image_qa(model, processor, image_path, question):
    # 载入图像
    image = load_image(image_path)
    if image is None:
        print("No image found. Exiting.")
        return

    # 提取图像特征
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # 将问题转化为输入 ID
    text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(input_ids=text_inputs, pixel_values=pixel_values)
    
    # 从输出中获取生成的答案
    generated_answer = processor.decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)

    return generated_answer

# 测试模型
generated_answer = test_image_qa(model, processor, test_image_path, test_question)

print(f"Question: {test_question}")
print(f"Generated Answer: {generated_answer}")
