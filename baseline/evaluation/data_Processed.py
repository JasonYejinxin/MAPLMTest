import os
import json
import torch
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

# 数据路径
image_folder = "/path/to/images"  # 图片主文件夹路径
qa_json_path = "/path/to/qa.json"  # 文本问答路径
output_dir = "/path/to/output_dir"  # 模型保存路径

# 加载文本问答数据
with open(qa_json_path, "r") as f:
    qa_data = json.load(f)  # 假设这是一个列表，每个元素是{"answer": "...", "frame": "...", "question": "..."}

# 加载 BLIP2 的特征提取器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# 加载 T5 模型
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl").to(device)

# 数据集类
class ImageTextDataset(Dataset):
    def __init__(self, image_folder, qa_data, processor, tokenizer):
        self.image_folder = image_folder
        self.qa_data = qa_data
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        # 获取对应的问答数据
        qa_entry = self.qa_data[idx]
        frame = qa_entry["frame"]
        question = qa_entry["question"]
        answer = qa_entry["answer"]

        # 加载对应的图片
        images = []
        image_dir = os.path.join(self.image_folder, frame)
        for img_file in sorted(os.listdir(image_dir)):  # 确保图片按顺序加载
            if img_file.endswith((".jpg", ".png")):
                img_path = os.path.join(image_dir, img_file)
                images.append(Image.open(img_path).convert("RGB"))

        # 提取图片特征
        image_features = []
        for image in images:
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = blip_model.vision_model(**inputs)
            image_features.append(outputs.last_hidden_state.mean(dim=1).squeeze(0))  # 全局特征

        # 拼接图片特征
        image_features = torch.cat(image_features, dim=0)

        # 编码文本
        input_ids = self.tokenizer.encode(question, return_tensors="pt", padding=True, truncation=True).squeeze(0)
        target_ids = self.tokenizer.encode(answer, return_tensors="pt", padding=True, truncation=True).squeeze(0)

        return image_features, input_ids, target_ids

# 初始化数据集和数据加载器
dataset = ImageTextDataset(image_folder, qa_data, blip_processor, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 优化器和训练设置
optimizer = torch.optim.AdamW(t5_model.parameters(), lr=5e-5)
epochs = 5

# 训练循环
for epoch in range(epochs):
    t5_model.train()
    for batch_idx, (image_features, input_ids, target_ids) in enumerate(dataloader):
        optimizer.zero_grad()

        # 模型输入
        outputs = t5_model(
            input_ids=input_ids.to(device),
            attention_mask=(input_ids != tokenizer.pad_token_id).to(device),
            labels=target_ids.to(device),
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # 打印进度
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

# 保存模型
t5_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

