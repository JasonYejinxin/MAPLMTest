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




import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else' cpu')

# 初始化 BLIP 处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", ignore_mismatched_sizes=True)

model = model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 假设qa.json文件是存储在当前目录下
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTrain.json"  # 替换为你的qa.json文件路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

# 载入图像（每个frame文件夹只加载第一张图片）
def load_images_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train", frame_id)  # 指定图片路径

    # 检查文件夹是否存在
    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None  # 返回None，表示该frame没有对应的图片

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:1]:  # 只加载第一张图片
        img_path = os.path.join(frame_path, img_name)
        print(img_path)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None  # 如果没有图片，返回None

# 处理图像和文本数据
def process_multimodal_data(qa_data):
    inputs = []
    targets = []

    for data in qa_data:
        question = data["question"]
        frame = data["frame"]
        answer = data["answer"]

        # 加载该 frame 下的第一张图片
        images = load_images_from_frame(frame)
        
        # 如果没有加载到图像，跳过该条数据
        if images is None:
            continue
        
        # 提取图像特征
        pixel_values = processor(images=images[0], return_tensors="pt").pixel_values.to("cuda")
        
        # 将问题转化为输入 ID
        text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

        # 将答案转化为目标 ID (作为监督学习的标签)
        target_ids = processor(text=answer, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

        # 保存输入和目标
        inputs.append((text_inputs, pixel_values))
        targets.append(target_ids)

    return inputs, targets

# 训练模型
def train_model(qa_data, epochs=5, save_interval=1):
    inputs, targets = process_multimodal_data(qa_data)

    # 训练循环
    for epoch in range(epochs):  # 假设训练 5 个 epoch
        print(f"round of traning epochs is number{epoch}")
        model.train()
        for i in range(len(inputs)):
            input_ids, pixel_values = inputs[i]
            target_ids = targets[i]
            
            # print(f"pixel shape:{pixel_values},input_ids shape:{input_ids},labels shape:{target_ids}")
            # # 向模型传递输入和目标
            # outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=target_ids)
        try:
            outputs = model(input_ids=input_ids, labels=target_ids, pixel_values=pixel_values)
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"input_ids shape: {input_ids.shape}")
            print(f"pixel_values shape: {pixel_values.shape}")
            print(f"labels shape: {target_ids.shape}")
        

            loss = outputs.loss
            loss.backward()

            # 更新权重
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

        # 每个epoch结束后保存模型和处理器
        model_save_path = f"./blip2_flan_t5_epoch_{epoch+1}"
        processor_save_path = f"./blip2_flan_t5_epoch_{epoch+1}"

        model.save_pretrained(model_save_path)
        processor.save_pretrained(processor_save_path)

        print(f"Model and processor saved for epoch {epoch+1} at {model_save_path}.")

# 启动训练
train_model(qa_data)




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



# 将数据组织为一个 Dataset 对象
class MultimodalDataset(Dataset):
    def __init__(self, qa_data, processor):
        self.qa_data = qa_data
        self.processor = processor
        self.inputs, self.targets = process_multimodal_data(self.qa_data)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids, pixel_values = self.inputs[idx]
        target_ids = self.targets[idx]
        return input_ids, pixel_values, target_ids

# 训练模型
def train_model(qa_data, epochs=5, save_interval=1, batch_size=4):
    # 通过 DataLoader 进行批量数据加载
    dataset = MultimodalDataset(qa_data, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(epochs):  # 假设训练 5 个 epoch
        print(f"Round of training epoch {epoch + 1}")
        model.train()
        
        for i, (input_ids, pixel_values, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            pixel_values = pixel_values.to(device)
            target_ids = target_ids.to(device)
            
            try:
                # 向模型传递输入和目标
                outputs = model(input_ids=input_ids, labels=target_ids, pixel_values=pixel_values)
                loss = outputs.loss
                loss.backward()

                # 更新权重
                optimizer.step()
                optimizer.zero_grad()

                print(f"Batch {i + 1} - Loss: {loss.item()}")
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"pixel_values shape: {pixel_values.shape}")
                print(f"labels shape: {target_ids.shape}")
                
        print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")




rror occurred: too many values to unpack (expected 4)
input_ids shape: torch.Size([4, 1, 12])
pixel_values shape: torch.Size([4, 1, 3, 224, 224])
labels shape: torch.Size([4, 1, 5])
Error occurred: too many values to unpack (expected 4)
input_ids shape: torch.Size([4, 1, 12])
pixel_values shape: torch.Size([4, 1, 3, 224, 224])
labels shape: torch.Size([4, 1, 5])
Error occurred: too many values to unpack (expected 4)
input_ids shape: torch.Size([4, 1, 12])
pixel_values shape: torch.Size([4, 1, 3, 224, 224])
labels shape: torch.Size([4, 1, 5])
Error occurred: too many values to unpack (expected 4)
input_ids shape: torch.Size([4, 1, 12])
pixel_values shape: torch.Size([4, 1, 3, 224, 224])
labels shape: torch.Size([4, 1, 5])
Error occurred: too many values to unpack (expected 4)
input_ids shape: torch.Size([4, 1, 12])
pixel_values shape: torch.Size([4, 1, 3, 224, 224])
labels shape: torch.Size([4, 1, 5])
Error occurred: too many values to unpack (expected 4)
input_ids shape: torch.Size([4, 1, 12])
pixel_values shape: torch.Size([4, 1, 3, 224, 224])
labels shape: torch.Size([4, 1, 5])
Error occurred: too many values to unpack (expected 4)
input_ids shape: torch.Size([4, 1, 12])
pixel_values shape: torch.Size([4, 1, 3, 224, 224])
labels shape: torch.Size([4, 1, 5])
Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/1image_dataProcessed.py", line 138, in <module>
    train_model(qa_data)
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/1image_dataProcessed.py", line 104, in train_model
    for i, (input_ids, pixel_values, target_ids) in enumerate(dataloader):
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 757, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/utils/data/_utils/collate.py", line 398, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/utils/data/_utils/collate.py", line 212, in collate
    collate(samples, collate_fn_map=collate_fn_map)
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/utils/data/_utils/collate.py", line 155, in collate
    return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/utils/data/_utils/collate.py", line 272, in collate_tensor_fn
    return torch.stack(batch, 0, out=out)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: stack expects each tensor to be equal size, but got [1, 5] at entry 0 and [1, 8] at entry 1










import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 BLIP 处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", ignore_mismatched_sizes=True)
model = model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 假设 qa.json 文件是存储在当前目录下
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTrain.json"  # 替换为你的路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

# 载入图像（每个 frame 文件夹只加载第一张图片）
def load_images_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train", frame_id)  # 指定图片路径

    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:1]:  # 只加载第一张图片
        img_path = os.path.join(frame_path, img_name)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None

# 处理图像和文本数据
def process_multimodal_data(qa_data, max_question_len=32, max_answer_len=16):
    inputs = []
    targets = []

    for data in qa_data:
        question = data["question"]
        frame = data["frame"]
        answer = data["answer"]

        images = load_images_from_frame(frame)
        if images is None:
            continue

        pixel_values = processor(images=images[0], return_tensors="pt").pixel_values
        text_inputs = processor(
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_question_len
        ).input_ids
        target_ids = processor(
            text=answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_answer_len
        ).input_ids

        inputs.append((text_inputs, pixel_values))
        targets.append(target_ids)

    return inputs, targets

# 自定义 collate_fn 解决张量堆叠问题
def collate_fn(batch):
    input_ids, pixel_values, target_ids = zip(*batch)
    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.cat(pixel_values, dim=0)
    target_ids = torch.cat(target_ids, dim=0)
    return input_ids, pixel_values, target_ids

# 构造 Dataset 类
class MultimodalDataset(Dataset):
    def __init__(self, qa_data, processor):
        self.inputs, self.targets = process_multimodal_data(qa_data)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx][0], self.inputs[idx][1], self.targets[idx]

# 训练模型
def train_model(qa_data, epochs=5, save_interval=1, batch_size=4):
    dataset = MultimodalDataset(qa_data, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        print(f"Round of training epoch {epoch + 1}")
        model.train()

        for i, (input_ids, pixel_values, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            pixel_values = pixel_values.to(device)
            target_ids = target_ids.to(device)

            try:
                outputs = model(input_ids=input_ids, labels=target_ids, pixel_values=pixel_values)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                print(f"Batch {i + 1} - Loss: {loss.item()}")
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"pixel_values shape: {pixel_values.shape}")
                print(f"labels shape: {target_ids.shape}")

        print(f"Epoch {epoch + 1} completed.")

        model_save_path = f"./blip2_flan_t5_epoch_{epoch + 1}"
        processor_save_path = f"./blip2_flan_t5_epoch_{epoch + 1}"

        model.save_pretrained(model_save_path)
        processor.save_pretrained(processor_save_path)

        print(f"Model and processor saved for epoch {epoch + 1} at {model_save_path}.")

# 启动训练
train_model(qa_data)


import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 BLIP 处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", ignore_mismatched_sizes=True)

model = model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 假设qa.json文件是存储在当前目录下
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTrain.json"  # 替换为你的qa.json文件路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

# 载入图像（每个frame文件夹只加载第一张图片）
def load_images_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train", frame_id)  # 指定图片路径

    # 检查文件夹是否存在
    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None  # 返回None，表示该frame没有对应的图片

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:1]:  # 只加载第一张图片
        img_path = os.path.join(frame_path, img_name)
        print(img_path)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None  # 如果没有图片，返回None

# 自定义Dataset类
class MultiModalDataset(Dataset):
    def __init__(self, qa_data, processor):
        self.qa_data = qa_data
        self.processor = processor

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        data = self.qa_data[idx]
        question = data["question"]
        frame = data["frame"]
        answer = data["answer"]

        # 加载该 frame 下的第一张图片
        images = load_images_from_frame(frame)
        if images is None:
            return None  # 返回None，如果没有图片

        # 提取图像特征
        pixel_values = self.processor(images=images[0], return_tensors="pt").pixel_values.squeeze().to(device)
        
        # 将问题转化为输入 ID
        text_inputs = self.processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze().to(device)

        # 将答案转化为目标 ID (作为监督学习的标签)
        target_ids = self.processor(text=answer, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze().to(device)

        return text_inputs, pixel_values, target_ids

# 训练模型
def train_model(qa_data, epochs=5, batch_size=4, save_interval=1):
    dataset = MultiModalDataset(qa_data, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(epochs):  # 假设训练 5 个 epoch
        print(f"round of traning epochs is number {epoch}")
        model.train()
        for batch_idx, (input_ids, pixel_values, target_ids) in enumerate(dataloader):
            if input_ids is None or pixel_values is None or target_ids is None:
                continue  # 跳过没有加载到数据的batch
            
            try:
                # 传递给模型
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=target_ids)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"pixel_values shape: {pixel_values.shape}")
                print(f"labels shape: {target_ids.shape}")

            loss = outputs.loss
            loss.backward()

            # 更新权重
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

        # 每个epoch结束后保存模型和处理器
        model_save_path = f"./blip2_flan_t5_epoch_{epoch+1}"
        processor_save_path = f"./blip2_flan_t5_epoch_{epoch+1}"

        model.save_pretrained(model_save_path)
        processor.save_pretrained(processor_save_path)

        print(f"Model and processor saved for epoch {epoch+1} at {model_save_path}.")

# 启动训练
train_model(qa_data)



You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
round of traning epochs is number 0
/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train/FR8537/photo_forward.jpg
/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train/FR3354/photo_forward.jpg
/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train/FR6113/photo_forward.jpg
/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train/FR3624/photo_forward.jpg
Error occurred: Expected input batch_size (44) to match target batch_size (16).
input_ids shape: torch.Size([4, 12])
pixel_values shape: torch.Size([4, 3, 224, 224])
labels shape: torch.Size([4, 5])
Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/oneImageTest.py", line 117, in <module>
    train_model(qa_data)
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/oneImageTest.py", line 98, in train_model
    loss = outputs.loss
           ^^^^^^^


import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 BLIP 处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", ignore_mismatched_sizes=True)

model = model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 假设qa.json文件是存储在当前目录下
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTrain.json"  # 替换为你的qa.json文件路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

# 载入图像（每个frame文件夹只加载第一张图片）
def load_images_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train", frame_id)  # 指定图片路径

    # 检查文件夹是否存在
    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None  # 返回None，表示该frame没有对应的图片

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:1]:  # 只加载第一张图片
        img_path = os.path.join(frame_path, img_name)
        print(img_path)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None  # 如果没有图片，返回None

# 自定义Dataset类
class MultiModalDataset(Dataset):
    def __init__(self, qa_data, processor):
        self.qa_data = qa_data
        self.processor = processor

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        data = self.qa_data[idx]
        question = data["question"]
        frame = data["frame"]
        answer = data["answer"]

        # 加载该 frame 下的第一张图片
        images = load_images_from_frame(frame)
        if images is None:
            return None  # 返回None，如果没有图片

        # 提取图像特征
        pixel_values = self.processor(images=images[0], return_tensors="pt").pixel_values.squeeze().to(device)
        
        # 将问题转化为输入 ID
        text_inputs = self.processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze().to(device)

        # 将答案转化为目标 ID (作为监督学习的标签)
        target_ids = self.processor(text=answer, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze().to(device)

        return text_inputs, pixel_values, target_ids

# 训练模型
def train_model(qa_data, epochs=5, batch_size=4, save_interval=1):
    dataset = MultiModalDataset(qa_data, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(epochs):  # 假设训练 5 个 epoch
        print(f"round of traning epochs is number {epoch}")
        model.train()
        for batch_idx, (input_ids, pixel_values, target_ids) in enumerate(dataloader):
            if input_ids is None or pixel_values is None or target_ids is None:
                continue  # 跳过没有加载到数据的batch

            # 检查每个batch的大小是否一致
            print(f"Batch {batch_idx} - input_ids shape: {input_ids.shape}, pixel_values shape: {pixel_values.shape}, target_ids shape: {target_ids.shape}")
            
            # 检查批次大小是否一致
            assert input_ids.shape[0] == pixel_values.shape[0] == target_ids.shape[0], \
                f"Mismatch in batch sizes: {input_ids.shape[0]} vs {pixel_values.shape[0]} vs {target_ids.shape[0]}"
            
            try:
                # 传递给模型
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=target_ids)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"pixel_values shape: {pixel_values.shape}")
                print(f"labels shape: {target_ids.shape}")

            loss = outputs.loss
            loss.backward()

            # 更新权重
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

        # 每个epoch结束后保存模型和处理器
        model_save_path = f"./blip2_flan_t5_epoch_{epoch+1}"
        processor_save_path = f"./blip2_flan_t5_epoch_{epoch+1}"

        model.save_pretrained(model_save_path)
        processor.save_pretrained(processor_save_path)

        print(f"Model and processor saved for epoch {epoch+1} at {model_save_path}.")

# 启动训练
train_model(qa_data)
Batch 0 - input_ids shape: torch.Size([4, 12]), pixel_values shape: torch.Size([4, 3, 224, 224]), target_ids shape: torch.Size([4, 5])
Error occurred: Expected input batch_size (44) to match target batch_size (16).
input_ids shape: torch.Size([4, 12])
pixel_values shape: torch.Size([4, 3, 224, 224])
labels shape: torch.Size([4, 5])
Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/oneImageTest.py", line 124, in <module>
    train_model(qa_data)
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/oneImageTest.py", line 105, in train_model
    loss = outputs.loss
           ^^^^^^^
UnboundLocalError: cannot access local variable 'outputs' where it is not associated with a value





# 确保 target_ids 长度与问题输入相同
        target_length = text_inputs.size(1)  # 获取问题的长度
        target_ids = target_ids[:, :target_length]  # 截断目标序列长度以匹配问题长度
        if target_ids.size(1) < target_length:
            # 如果目标长度小于问题长度，进行填充
            pad_length = target_length - target_ids.size(1)
            padding = torch.full((target_ids.size(0), pad_length), processor.tokenizer.pad_token_id).to(device)
            target_ids = torch.cat((target_ids, padding), dim=1)
