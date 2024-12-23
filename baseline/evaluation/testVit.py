from transformers import ViTFeatureExtractor, ViTModel
import torch
from PIL import Image
import os
import re
import json

# 自定义排序函数，提取名称中的数字部分
def extract_number(name):
    match = re.search(r'\d+', name)  # 匹配文件夹名中的数字部分
    return int(match.group()) if match else float('inf')  # 如果没有数字，放到最后


# 加载预训练模型
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name)

# 指定图片文件夹路径
data_folder = "/Users/jinxinye/Desktop/python/Project/MAPLM/baseline/evaluation/data/maplm_v0.1/train"

# 获取按数字部分排序后的文件夹列表
sorted_folders = sorted(os.listdir(data_folder), key=extract_number)

# 初始化保存特征向量的数组
features_list = []

#最终的文本-数据对
data_pairs = []
print("start")
# 遍历主文件夹中的所有子文件夹
for folder_name in sorted_folders:
    folder_path = os.path.join(data_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"processing folder:{folder_name}" )
        # 获取该子文件夹中的所有图片路径
        image_paths = sorted([os.path.join(folder_path, f) 
                              for f in os.listdir(folder_path) 
                              if f.endswith(('.jpg', '.png'))])

        if len(image_paths) != 4:
            print(f"Skipping folder {folder_name}: expected 4 images, found {len(image_paths)}")
            continue

        # 读取图片并提取特征
        image_features = []
        for image_path in image_paths:
            image = Image.open(image_path)

            # 确保图片是 RGB 格式
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 预处理并提取 ViT 特征
            inputs = feature_extractor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = vit_model(**inputs)

            # 提取全局特征向量
            feature = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            image_features.append(feature.squeeze(0).tolist())

        # 将 4 张图片的特征拼接
        # concatenated_features = torch.cat(image_features, dim=0)  # Shape: [4 * hidden_size]
        concatenated_features = torch.cat([torch.tensor(f) for f in image_features], dim=0).tolist()  # [4 * hidden_size]


        # 假设对应的文本是文件夹名
        corresponding_text = folder_name

        # 保存 (图片特征, 文本) 数据对
        # data_pairs.append((concatenated_features, corresponding_text))
        data_pairs.append({
            "features": concatenated_features,  # 3072 长度的特征向量
            "text": corresponding_text
        })

    # else: 
    #     image_features = []
    
    #     image = Image.open(folder_path)

    #         # 确保图片是 RGB 格式
    #     if image.mode != "RGB":
    #         image = image.convert("RGB")

    #         # 预处理并提取 ViT 特征
    #     inputs = feature_extractor(images=image, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = vit_model(**inputs)

    #         # 提取全局特征向量
    #     feature = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
    #     image_features.append(feature.squeeze(0).tolist())

    #     # 将 4 张图片的特征拼接
    #     # concatenated_features = torch.cat(image_features, dim=0)  # Shape: [4 * hidden_size]
    #     concatenated_features = torch.cat([torch.tensor(f) for f in image_features], dim=0).tolist()  # [4 * hidden_size]

    #     # 假设对应的文本是文件夹名
    #     corresponding_text = folder_name

    #     # 保存 (图片特征, 文本) 数据对
    #     # data_pairs.append((concatenated_features, corresponding_text))
    #     data_pairs.append({
    #         "features": concatenated_features,  # 3072 长度的特征向量
    #         "text": corresponding_text
    #     })


# # 查看结果
# print(f"Processed {len(data_pairs)} folders")
# print(f"Example feature shape: {data_pairs[0][0].shape}, Example text: {data_pairs[0][1]}")

# 保存为 JSON 文件
output_file = "output_data.json"
with open(output_file, "w") as f:
    f.write("[\n")
    for i, pair in enumerate(data_pairs):
        json.dump(pair, f)
        if i < len(data_pairs) - 1:
            f.write(",\n")  # 在每个对象后添加逗号，最后一个对象除外
    f.write("\n]")
print(f"Saved {len(data_pairs)} data pairs to {output_file}")
