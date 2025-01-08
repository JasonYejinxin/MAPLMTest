import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Blip2Processor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

# 设置 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据准备类
class ImageTextDataset(Dataset):
    def __init__(self, img_dir, qa_json_path, processor, tokenizer, max_length=512):
        self.img_dir = img_dir
        self.qa_data = self.load_qa_data(qa_json_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_qa_data(self, qa_json_path):
        with open(qa_json_path, 'r') as f:
            qa_data = json.load(f)
        return qa_data

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        question = qa['question']
        answer = qa['answer']
        image_paths = qa['image_paths']  # 从QA中获取图片路径

        # 提取图片特征
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        image_features = inputs.pixel_values.squeeze(0)  # Shape: [4, hidden_size] (4 images)

        # Tokenize question and answer
        input_text = question
        target_text = answer
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        targets = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)

        return {
            'image_features': image_features,
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'labels': targets.input_ids.squeeze(0),
        }

# 配置数据路径
img_dir = "/path/to/images"  # 替换为你的图片文件夹路径
qa_json_path = "/path/to/qa.json"  # 替换为你的 qa.json 文件路径

# 处理器和tokenizer
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# 创建数据集和数据加载器
dataset = ImageTextDataset(img_dir, qa_json_path, processor, tokenizer)
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 加载模型
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# 设置训练参数
epochs = 5
learning_rate = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        # 获取批次数据
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        image_features = batch['image_features'].to(device)

        # 模型前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播并更新权重
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader)}")

    # 每个epoch保存模型
    model.save_pretrained(f"model_epoch_{epoch+1}")
    tokenizer.save_pretrained(f"model_epoch_{epoch+1}")
