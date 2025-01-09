import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, BlipModel, T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW

# 数据集类
class MultimodalDataset(Dataset):
    def __init__(self, qa_file, image_folder, processor):
        self.image_folder = image_folder
        self.processor = processor
        with open(qa_file, 'r') as f:
            self.qa_data = json.load(f)
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa_item = self.qa_data[idx]
        frame = qa_item["frame"]
        question = qa_item["question"]
        answer = qa_item["answer"]

        # 读取 4 张图片
        frame_folder = os.path.join(self.image_folder, frame)
        image_paths = sorted([os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith(('.jpg', '.png'))])
        if len(image_paths) != 4:
            raise ValueError(f"Frame {frame} does not contain exactly 4 images.")
        
        # 预处理图片
        images = [self.processor.image_processor.open(image_path) for image_path in image_paths]
        pixel_values = self.processor(images=images, return_tensors="pt", padding=True)["pixel_values"]

        return {
            "pixel_values": pixel_values,
            "question": question,
            "answer": answer
        }

# 数据预处理
class MultimodalCollator:
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Tokenize question and answer
        question_encodings = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=128)
        answer_encodings = self.tokenizer(answers, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": question_encodings["input_ids"],
            "attention_mask": question_encodings["attention_mask"],
            "labels": answer_encodings["input_ids"]
        }

# 模型定义
class MultimodalModel(torch.nn.Module):
    def __init__(self, vision_model, qformer, text_model):
        super().__init__()
        self.vision_model = vision_model
        self.qformer = qformer
        self.text_model = text_model
    
    def forward(self, pixel_values, input_ids, attention_mask, labels):
        # 图像特征提取
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state

        # Q-Former 融合
        multimodal_features = self.qformer(inputs_embeds=image_features, decoder_input_ids=input_ids, attention_mask=attention_mask)

        # 语言模型生成
        outputs = self.text_model(input_ids=multimodal_features, labels=labels)
        return outputs.loss, outputs.logits

# 设置路径
qa_file = "/Users/jinxinye/Desktop/python/Project/MAPLM/baseline/evaluation/qa.json"  # 替换为你的 qa.json 路径
image_folder = "/Users/jinxinye/Desktop/python/Project/MAPLM/baseline/evaluation/data/maplm_v0.1/train"  # 替换为你的图片目录路径

# 初始化组件
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
text_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

vision_model = processor.model.vision_model  # 使用 BLIP2 的视觉模型
qformer = processor.model.qformer  # 使用 BLIP2 的 Q-Former

model = MultimodalModel(vision_model=vision_model, qformer=qformer, text_model=text_model)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据集和加载器
dataset = MultimodalDataset(qa_file=qa_file, image_folder=image_folder, processor=processor)
collator = MultimodalCollator(processor=processor, tokenizer=tokenizer)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

# 优化器
optimizer = AdamW(model.parameters(), lr=1e-4)

# 训练循环
epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in data_loader:
        pixel_values = batch["pixel_values"].to("cuda")
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")

        optimizer.zero_grad()
        loss, _ = model(pixel_values, input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader)}")

# 保存模型
torch.save(model.state_dict(), "multimodal_model.pth")
print("Model saved to multimodal_model.pth")
