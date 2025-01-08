from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import ViTFeatureExtractor, ViTModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from PIL import Image

# 1. 加载模型和预训练的特征提取器
vit_model_name = "google/vit-base-patch16-224-in21k"  # 或使用更大的模型
flan_t5_model_name = "Salesforce/blip2-flan-t5-xl"  # FLAN-T5-XL模型

# 加载ViT模型和提取器
feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)
vit_model = ViTModel.from_pretrained(vit_model_name)

# 加载FLAN-T5-XL模型和Tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_model_name)

# 2. 自定义Dataset类，处理图像和问答数据
class ImageTextDataset(Dataset):
    def __init__(self, data_folder, tokenizer, feature_extractor):
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for folder_name in os.listdir(self.data_folder):
            folder_path = os.path.join(self.data_folder, folder_name)
            if os.path.isdir(folder_path):
                # 读取图像
                image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
                if len(image_paths) != 4:  # 每组图像四张
                    continue

                # 读取问题和答案
                qa_path = os.path.join(folder_path, "qa.json")
                with open(qa_path, "r") as f:
                    qa_data = json.load(f)

                for qa in qa_data['qa']:
                    question = qa['question']
                    answer = qa['answer']
                    data.append((image_paths, question, answer))

        return data

    def __getitem__(self, idx):
        image_paths, question, answer = self.data[idx]
        # 处理图像特征
        images = [Image.open(img_path) for img_path in image_paths]
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)  # [batch_size, seq_len, hidden_size]

        # 处理问题文本
        question_input = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = question_input.input_ids.squeeze(0)
        attention_mask = question_input.attention_mask.squeeze(0)

        # 处理答案文本
        answer_input = self.tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=128)
        target_ids = answer_input.input_ids.squeeze(0)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }

    def __len__(self):
        return len(self.data)

# 3. 数据加载器
data_folder = "/Users/jinxinye/Desktop/python/Project/MAPLM/baseline/evaluation/data/maplm_v0.1/train"  # 替换为数据目录
dataset = ImageTextDataset(data_folder, t5_tokenizer, feature_extractor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 4. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)

optimizer = torch.optim.AdamW(t5_model.parameters(), lr=5e-5)

# 训练循环
epochs = 3
for epoch in range(epochs):
    t5_model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # ViT 特征提取
        with torch.no_grad():
            vit_outputs = vit_model(pixel_values=pixel_values)
            image_features = vit_outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]

        # FLAN-T5 输入
        input_ids = torch.cat([input_ids, image_features], dim=1)  # 结合图像和文字输入
        attention_mask = torch.cat([attention_mask, torch.ones_like(image_features)], dim=1)

        # 模型输出
        outputs = t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# 5. 保存模型
t5_model.save_pretrained("/path/to/save/your/model")
t5_tokenizer.save_pretrained("/path/to/save/your/tokenizer")
