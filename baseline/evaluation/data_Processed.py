import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
import json

# 数据路径
image_folder = "/path/to/your/images"  # 包含 FR 文件夹的路径
qa_file = "/path/to/your/qa.json"     # 问答文件路径
output_file = "/path/to/output.json"  # 输出的 JSON 文件路径
output_dir = "/path/to/save/fine_tuned_model"
# 加载问答数据
with open(qa_file, "r") as f:
    qa_data = json.load(f)

# 加载 BLIP2 模型
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-base")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-base")

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化结果字典
output_data = []

# 遍历问答数据
for qa_entry in qa_data:
    frame_id = qa_entry["frame"]
    question = qa_entry["question"]
    answer = qa_entry["answer"]
    
    # 构造对应的图片路径
    frame_folder = os.path.join(image_folder, frame_id)
    if not os.path.exists(frame_folder):
        print(f"Warning: Frame folder {frame_folder} does not exist. Skipping.")
        continue
    
    # 加载图片并提取特征
    image_features = []
    for img_name in sorted(os.listdir(frame_folder)):
        img_path = os.path.join(frame_folder, img_name)
        if not img_path.lower().endswith((".jpg", ".png")):
            continue
        
        # 加载图片
        image = Image.open(img_path).convert("RGB")
        
        # 图像处理并提取特征
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.vision_model(**inputs)
            pooled_feature = features.pooler_output.squeeze(0).cpu().tolist()  # 提取池化特征
            image_features.append(pooled_feature)
    
    # 检查图片数量是否为 4
    if len(image_features) != 4:
        print(f"Skipping {frame_id}: Expected 4 images, found {len(image_features)}.")
        continue
    
    # 合并图片特征
    concatenated_features = torch.tensor(image_features).flatten().tolist()

    # 保存到结果中
    output_data.append({
        "frame": frame_id,
        "question": question,
        "answer": answer,
        "image_features": concatenated_features
    })

# 将结果保存为 JSON 文件
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Data saved to {output_file}.")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

