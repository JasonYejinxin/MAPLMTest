import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型和处理器
model_path = "./blip2_flan_t5_epoch_5_concatenate"  # 替换为保存的模型路径
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model = model.to(device)
model.eval()

# 加载图片 (推理时加载单帧图像)
def load_images_for_inference(frame_path):
    images = []
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Frame folder {frame_path} not found.")
    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:4]:  # 加载最多 4 张图片
        img_path = os.path.join(frame_path, img_name)
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    if not images:
        raise ValueError(f"No valid images found in {frame_path}.")
    return images

# 特征拼接（与训练保持一致）
def concatenate_image_features(images):
    pixel_values = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").pixel_values.to(device)
        pixel_values.append(inputs.to(device))
    pixel_values = torch.cat(pixel_values, dim=1)  # [1, 12, H, W]
    return pixel_values

# 推理函数
def generate_answer(frame_path, question, feature_method="concatenate"):
    # 加载图片
    images = load_images_for_inference(frame_path)

    # 特征处理
    if feature_method == "concatenate":
        pixel_values = concatenate_image_features(images)
    else:
        raise ValueError("Only 'concatenate' feature method is supported in this inference code.")

    # 处理问题
    text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    # 模型生成答案
    with torch.no_grad():
        outputs = model.generate(input_ids=text_inputs, pixel_values=pixel_values, max_length=50)
        answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

# 示例推理
if __name__ == "__main__":
    frame_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/test"  # 替换为测试帧的文件夹路径
    question = "How many lanes in current road?"  # 替换为你想问的问题

    try:
        answer = generate_answer(frame_path, question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error occurred during inference: {e}")


Error occurred during inference: No valid images found in /home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/test.




