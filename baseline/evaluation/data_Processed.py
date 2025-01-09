import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AdamW,
)

# 路径设置
image_folder = "./train"  # 图片的主文件夹路径
qa_json_path = "./qa.json"  # QA 数据的路径
model_save_path = "./flan_t5_xl_multimodal.pth"  # 模型保存路径

# 加载模型和处理器
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 5

# 定义数据集
class MultimodalDataset(Dataset):
    def __init__(self, image_folder, qa_json_path, processor, tokenizer):
        self.image_folder = image_folder
        self.processor = processor
        self.tokenizer = tokenizer
        
        # 加载QA数据
        with open(qa_json_path, "r") as f:
            self.qa_data = json.load(f)

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        qa_entry = self.qa_data[idx]
        frame = qa_entry["frame"]
        question = qa_entry["question"]
        answer = qa_entry["answer"]

        # 加载该 frame 中的所有图像
        frame_folder = os.path.join(self.image_folder, frame)
        image_paths = sorted(
            [os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith((".jpg", ".png"))]
        )
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

        # 提取图像特征
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vision_outputs = blip_model.vision_model(**inputs)
        image_features = vision_outputs.last_hidden_state.mean(dim=1)

        # 处理文本数据
        text_inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        ).to(device)
        labels = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        ).input_ids.to(device)

        return image_features, text_inputs.input_ids, text_inputs.attention_mask, labels


# 创建数据集和 DataLoader
dataset = MultimodalDataset(image_folder, qa_json_path, blip_processor, t5_tokenizer)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义优化器
optimizer = AdamW(t5_model.parameters(), lr=5e-5)

# 模型训练
t5_model.to(device)
t5_model.train()

for epoch in range(epochs):
    total_loss = 0
    for image_features, input_ids, attention_mask, labels in data_loader:
        # 将图像特征拼接到文本前面
        encoder_inputs = torch.cat([image_features, input_ids], dim=1)

        # 前向传播
        outputs = t5_model(
            input_ids=encoder_inputs,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# 保存模型
torch.save(t5_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
