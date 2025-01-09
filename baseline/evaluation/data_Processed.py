import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, BlipProcessor
from PIL import Image
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类
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
        frame = qa['frame']
        
        # 获取该frame对应的图片文件夹路径
        image_paths = [os.path.join(self.img_dir, frame, f"{frame}_photo_{i+1}.jpg") for i in range(4)]
        
        # 一次性加载和处理所有图片
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        
        # 提取图像特征
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        image_features = inputs.pixel_values.squeeze(0)  # 形状: [4, hidden_size] (4张图片)

        # Tokenize问题和答案
        input_text = question
        target_text = answer
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        targets = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)

        return {
            'image_features': image_features,
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'labels': targets.input_ids.squeeze(0),
            'frame': frame,
            'question': question,
            'answer': answer
        }

# 加载模型和tokenizer
model_name = "t5-base"  # 可以更换为你需要的T5模型
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-base")

# 数据路径
img_dir = 'train'  # 图片存储路径
qa_json_path = 'qa.json'  # 问答json文件路径

# 创建数据集和数据加载器
dataset = ImageTextDataset(img_dir=img_dir, qa_json_path=qa_json_path, processor=processor, tokenizer=tokenizer)
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 设置优化器和损失函数
optimizer = torch.optim.AdamW(t5_model.parameters(), lr=5e-5)

# 训练函数
def train(model, dataloader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            image_features = batch['image_features'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

# 开始训练
train(t5_model, train_dataloader, optimizer, num_epochs=5)

# 保存训练好的模型
model_save_path = 'trained_model'
t5_model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

