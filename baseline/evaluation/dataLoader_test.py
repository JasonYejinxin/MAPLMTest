import argparse
import json
import random
from typing import List

from utils import load_data, load_data_trainOnly, load_data_gpt, get_result_file, acc_counter, compute_acc, load_model_output, retrieve_completion, completion_to_answer
#nltk library is used to concat string
from nltk.translate.bleu_score import sentence_bleu

#put paramaters into a container with parser, which is a object of Class argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/maplm_v0.1')
parser.add_argument('--model_output_file', type=str, default='maplm_test.json')
# parser.add_argument('--model_output_file', type=str, default='output.json')
parser.add_argument('--output_dir', type=str, default='runs')
parser.add_argument('--test_split', type=str, default='test')
parser.add_argument('--train_split', type=str, default='train')
parser.add_argument('--test_number', type=int, default=-1, help='Number of test frames to run (default: -1, all)')
parser.add_argument('--train_number', type=int, default=-1, help='Number of train frames to run (default: -1, all)')
parser.add_argument('--exp_label', type=str, default='exp_random', help='Experiment label')
parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')
#args form: jason
args = parser.parse_args()

results = dict(
    question_overall=acc_counter(),
    frame_overall=acc_counter(),
)

# #调用图片
# def load_image(image_file):
#     if image_file.startswith('http') or image_file.startswith('https'):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         image = Image.open(image_file).convert('RGB')
#     return image


#以下是执行文件
if __name__ == "__main__":
    # print('===== Input Arguments =====')
    # #把python对象args解码成python格式
    # print(json.dumps(vars(args), indent=4, sort_keys=True))
    #不知道是啥，应该是个随机数
    random.seed(args.random_seed)
    #分别赋值，不知道格式，根据方法有两个输出结果，problem和id，分别来自tool目录下的problems和pid_splits
    frames, frame_ids = load_data_trainOnly(args)
    #util有
    result_file_name = get_result_file(args)
    #遍历枚举 frame_ids，frame，frame_id是被赋予的value，frames是已经有的，是来自util的problem，有两个参数：image和qa
    # for i, frame_id in enumerate(frame_ids):
    #     print("=======start=======",i)
    #     #frames有一级index：FR1，二级index：image，qas
    #     frame = frames[frame_id]

    #     image = frame['image']
    #     qas = frame['qa']
    #     for j, qa in enumerate(qas):
    #         question = qa['question']
    #         if qa['task'] == 'closed choice':
    #             answer = qa['choices'][qa['answer']]
    #         else: answer = qa['answer']
    #         print(question + answer)


    i = 0
    for i, frame_id in enumerate(frame_ids):
        print("=======start=======",frame_ids[i])
        #frames有一级index：FR1，二级index：image，qas
        frame = frames[frame_id]
        image = frame['image']
        qas = frame['qa']
        for j, qa in enumerate(qas):
            question = qa['question']
            if question == "How many lanes in current road?":
                answer = qa['choices'][qa['answer']]
            else: continue
            print(question + answer)
    print(i)



        # corrects = []
        # #二级index，包含image和qas
        # for j, qa in enumerate(qas): 
        #     #逻辑在problem里面,如果不是选择题，则出循环
        #     if qa['task'] != 'closed choice':
        #         continue
        #     question = qa['question']
        #     choices: List[str] = qa['choices']
        #     true_answer: int = qa['answer']
        #     #随机数，0到所有选择任意一个
        #     random_guess: int = random.randint(0, len(choices) - 1)
        #     #如果二级坐标下，question数组中的value不在results字典里面，给这个key-map一个value：acc_couunter(),从0开始的计数器，意思是如果没有这个问题，则写入这个问题，从0开始记出现次数
        #     if question not in results:
        #         results[question] = acc_counter()
        #     #correct是布尔，corrects是数组，初始为空，每次循环加一个value,判断标准，random guess是否等于true answer，true是fr里面的，random是answer中任意一个
        #     correct = bool(random_guess == true_answer)
        #     corrects.append(correct)
        #     #results字典二维索引，question_overall，frame_overall是元素，total和correct在acc_counter里面，results[question]是每个question，value是出现次数
        #     results[question]['total'] += 1
        #     results[question]['correct'] += int(correct)
        #     results['question_overall']['total'] += 1
        #     results['question_overall']['correct'] += int(correct)
        
        # results['frame_overall']['total'] += 1
        # results['frame_overall']['correct'] += int(all(corrects))
    #输出结果，results是一个字典，
    # print('===== Results =====')

    # acc_dict = compute_acc(results)
    # print(json.dumps(acc_dict, indent=4, sort_keys=True))
    # print(json.dumps(results, indent=4, sort_keys=True))


Error occurred: Expected input batch_size (11) to match target batch_size (4).
input_ids shape: torch.Size([1, 12])
pixel_values shape: torch.Size([1, 3, 224, 224])
labels shape: torch.Size([1, 5])
Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/text.py", line 93, in train_model
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
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/text.py", line 120, in <module>
    train_model(qa_data)
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/text.py", line 101, in train_model
    loss = outputs.loss
           ^^^^^^^
UnboundLocalError: cannot access local variable 'outputs' where it is not associated with a value










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
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qa.json"  # 替换为你的qa.json文件路径
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
        
        # # 如果是单张图像，需要加上批次维度
        # pixel_values = pixel_values.unsqueeze(0)  # 添加一个batch维度，变成[1, channels, height, width]
        
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

