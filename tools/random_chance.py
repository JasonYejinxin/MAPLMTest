# Adapted from: https://github.com/lupantech/ScienceQA/blob/main/models/run_gpt3.py

import argparse
import json
import random
from typing import List

from utils import load_data, get_result_file, acc_counter, compute_acc
#把一系列参数 用parser这个argparse的对象 搞进args里面
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/maplm_v0.1')
parser.add_argument('--output_dir', type=str, default='runs')
parser.add_argument('--test_split', type=str, default='test')
parser.add_argument('--test_number', type=int, default=-1, help='Number of test frames to run (default: -1, all)')
parser.add_argument('--exp_label', type=str, default='exp_random', help='Experiment label')
parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')

args = parser.parse_args()
#结果是一个字典，util里面有，反正两个都是计数器
results = dict(
    question_overall=acc_counter(),
    frame_overall=acc_counter(),
)
#以下是执行文件
if __name__ == "__main__":
    print('===== Input Arguments =====')
    #把json对象args解码成python格式
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    #随机数
    random.seed(args.random_seed)
    #分别赋值，不知道格式，根据方法有两个输出结果，problem和id，分别来自tool目录下的problems和pid_splits
    frames, frame_ids = load_data(args)
    #util有
    result_file_name = get_result_file(args)
    #遍历枚举 frame_ids，frame，frame_id是被赋予的value，frames是已经有的，是来自util的problem，有两个参数：image和qa
    for i, frame_id in enumerate(frame_ids):
        #frames有一级index：FR1，二级index：image，qas
        frame = frames[frame_id]

        image = frame['image']
        qas = frame['qa']

        corrects = []
        #二级index，包含image和qas
        for j, qa in enumerate(qas): 
            #逻辑在problem里面,如果不是选择题，则出循环
            if qa['task'] != 'closed choice':
                continue
            question = qa['question']
            choices: List[str] = qa['choices']
            true_answer: int = qa['answer']
            #随机数，0到所有选择任意一个
            random_guess: int = random.randint(0, len(choices) - 1)
            #如果二级坐标下，question数组中的value不在results字典里面，给这个key-map一个value：acc_couunter(),从0开始的计数器，意思是如果没有这个问题，则写入这个问题，从0开始记出现次数
            if question not in results:
                results[question] = acc_counter()
            #correct是布尔，corrects是数组，初始为空，每次循环加一个value,判断标准，random guess是否等于true answer，true是fr里面的，random是answer中任意一个
            correct = bool(random_guess == true_answer)
            corrects.append(correct)
            #results字典二维索引，question_overall，frame_overall是元素，total和correct在acc_counter里面，results[question]是每个question，value是出现次数
            results[question]['total'] += 1
            results[question]['correct'] += int(correct)
            results['question_overall']['total'] += 1
            results['question_overall']['correct'] += int(correct)
        
        results['frame_overall']['total'] += 1
        results['frame_overall']['correct'] += int(all(corrects))
    #输出结果，results是一个字典，
    print('===== Results =====')
    acc_dict = compute_acc(results)
    print(json.dumps(acc_dict, indent=4, sort_keys=True))
    print(json.dumps(results, indent=4, sort_keys=True))
