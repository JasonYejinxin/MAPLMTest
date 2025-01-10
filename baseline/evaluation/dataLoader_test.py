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






Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/text.py", line 108, in <module>
    train_model(qa_data)
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/text.py", line 87, in train_model
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=target_ids)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/models/blip/modeling_blip.py", line 1109, in forward
    vision_outputs = self.vision_model(
                     ^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/models/blip/modeling_blip.py", line 724, in forward
    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/models/blip/modeling_blip.py", line 277, in forward
    batch_size, _, height, width = pixel_values.shape
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 4)


pixel shape:tensor([[[[[-1.0915, -1.0477, -1.0623,  ..., -1.2375, -1.1937, -1.1207],
           [-1.1353, -1.0623, -0.9893,  ..., -1.2083, -1.1061, -1.0769],
           [-1.1207, -1.0769, -1.0331,  ..., -1.1499, -1.1499, -1.1645],
           ...,
           [ 1.0982,  1.1420,  1.1858,  ...,  0.5727,  0.4997,  0.4559],
           [ 1.2150,  1.2296,  1.2880,  ...,  0.8501,  0.8063,  0.7771],
           [ 1.3464,  1.3756,  1.3610,  ...,  1.0252,  0.9814,  0.9814]],

          [[-0.8366, -0.7916, -0.8066,  ..., -0.9867, -0.9867, -1.0317],
           [-0.8816, -0.8066, -0.7316,  ..., -0.9717, -1.0467, -1.0467],
           [-0.8666, -0.8216, -0.7766,  ..., -0.9717, -1.0167, -1.0017],
           ...,
           [ 1.2194,  1.2645,  1.3095,  ...,  0.6792,  0.6041,  0.5441],
           [ 1.3395,  1.3545,  1.4145,  ...,  0.9643,  0.9193,  0.8893],
           [ 1.4746,  1.5046,  1.4896,  ...,  1.1444,  1.0994,  1.0994]],

          [[-0.4848, -0.4422, -0.4564,  ..., -0.5417, -0.6270, -0.6128],
           [-0.5275, -0.4564, -0.3853,  ..., -0.3995, -0.5417, -0.5986],
           [-0.5275, -0.4706, -0.4279,  ..., -0.3568, -0.5559, -0.5986],
           ...,
           [ 1.3354,  1.3922,  1.4207,  ...,  0.8234,  0.7523,  0.7097],
           [ 1.4491,  1.4633,  1.5202,  ...,  1.0936,  1.0510,  1.0225],
           [ 1.5771,  1.6055,  1.5913,  ...,  1.2643,  1.2216,  1.2216]]]]],
       device='cuda:0'),input_ids shape:tensor([[ 363,  773,   13, 1373, 3112,   19,   34,   16,    8, 1383,   58,    1]],
       device='cuda:0'),labels shape:tensor([[16612,   690,  1373,     5,     1]], device='cuda:0')








