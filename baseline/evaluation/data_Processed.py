import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# 自定义 Dataset 类
class QADataset(Dataset):
    def __init__(self, qa_data, frame_base_path, processor):
        self.qa_data = qa_data
        self.frame_base_path = frame_base_path
        self.processor = processor

    def __len__(self):
        return len(self.qa_data)

    def load_images_from_frame(self, frame_id):
        images = []
        frame_path = os.path.join(self.frame_base_path, frame_id)  # 指定图片路径
        if not os.path.exists(frame_path):
            print(f"Warning: Folder {frame_id} not found. Skipping this frame.")
            return None  # 如果文件夹不存在，返回 None
        for img_name in os.listdir(frame_path):
            img_path = os.path.join(frame_path, img_name)
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                img = Image.open(img_path).convert("RGB")
                images.append(img)
        return images

    def __getitem__(self, idx):
        data = self.qa_data[idx]
        question = data["question"]
        answer = data["answer"]
        frame_id = data["frame"]

        # 加载该 frame 下的图片
        images = self.load_images_from_frame(frame_id)
        if images is None:
            return None

        # 图像和文本数据处理
        pixel_values_list = []
        for img in images:
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
            pixel_values_list.append(pixel_values)

        input_ids = self.processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids
        labels = self.processor(text=answer, return_tensors="pt", padding=True, truncation=True).input_ids

        return input_ids.squeeze(0), labels.squeeze(0), torch.cat(pixel_values_list, dim=0).squeeze(0)  # 去掉 batch dimension

# collate_fn 函数，确保批次数据正确处理
def collate_fn(batch):
    # 过滤掉 None 数据
    batch = list(filter(lambda x: x is not None, batch))
    
    # 解包批次数据
    input_ids, labels, pixel_values = zip(*batch)
    
    # 将数据转换为适合批量处理的形式
    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    pixel_values = torch.stack(pixel_values, dim=0)

    return input_ids, labels, pixel_values

# 初始化 BLIP 处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", ignore_mismatched_sizes=True)

# 加载 QA 数据
qa_json_path = "/path/to/qa.json"  # 替换为你的qa.json文件路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

frame_base_path = "/Users/jinxinye/Desktop/python/Project/MAPLM/baseline/evaluation/data/maplm_v0.1/train"

# 创建数据集和 DataLoader
dataset = QADataset(qa_data, frame_base_path, processor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练循环
for epoch in range(5):  # 假设训练 5 个 epoch
    model.train()
    for batch in dataloader:
        input_ids, labels, pixel_values = batch
        input_ids = input_ids.to("cuda")
        labels = labels.to("cuda")
        pixel_values = pixel_values.to("cuda")

        # 向模型传递输入和目标
        outputs = model(input_ids=input_ids, labels=labels, pixel_values=pixel_values)

        loss = outputs.loss
        loss.backward()

        # 更新权重
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

    # 保存模型
    model.save_pretrained(f"./blip2_flan_t5_epoch_{epoch+1}")
    processor.save_pretrained(f"./blip2_flan_t5_epoch_{epoch+1}")

    print(f"Model and processor saved for epoch {epoch+1}.")








round of traning epochs is number0
Error occurred: Expected input batch_size (11) to match target batch_size (4).
input_ids shape: torch.Size([1, 12])
pixel_values shape: torch.Size([1, 3, 224, 224])
labels shape: torch.Size([1, 5])
Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/1image_dataProcessed.py", line 94, in train_model
    outputs = model(input_ids=input_ids, labels=target_ids, pixel_values=pixel_values)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/models/blip/modeling_blip.py", line 1119, in forward
    outputs = self.text_decoder(
              ^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/models/blip/modeling_blip_text.py", line 906, in forward
    lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/loss.py", line 1293, in forward
    return F.cross_entropy(
           ^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/functional.py", line 3479, in cross_entropy
    return torch._C._nn.cross_entropy_loss(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Expected input batch_size (11) to match target batch_size (4).

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/1image_dataProcessed.py", line 121, in <module>
    train_model(qa_data)
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/1image_dataProcessed.py", line 102, in train_model
    loss = outputs.loss
           ^^^^^^^
UnboundLocalError: cannot access local variable 'outputs' where it is not associated with a value















