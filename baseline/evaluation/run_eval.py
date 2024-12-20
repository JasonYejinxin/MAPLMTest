import argparse
import json
import random
from typing import List

from utils import load_data, load_data_gpt, get_result_file, acc_counter, compute_acc, load_model_output, retrieve_completion, \
    completion_to_answer
# nltk library is used to concat string
from nltk.translate.bleu_score import sentence_bleu
#put paramaters into a parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/maplm_v0.1')
parser.add_argument('--model_output_file', type=str, default='maplm_test.json')
# parser.add_argument('--model_output_file', type=str, default='output.json')
parser.add_argument('--output_dir', type=str, default='runs')
parser.add_argument('--test_split', type=str, default='test')
parser.add_argument('--test_number', type=int, default=-1, help='Number of test frames to run (default: -1, all)')
parser.add_argument('--exp_label', type=str, default='exp_random', help='Experiment label')
parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')
# Parser the parameters and returns a Namespace object
args = parser.parse_args()
#it will retrun a dict，key(question and frame)-value(counter), and the method is from utils，
results = dict(
    question_overall=acc_counter(),
    frame_overall=acc_counter(),
)

if __name__ == "__main__":
    print('===== Input Arguments =====')
    #decode args to python
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    
    random.seed(args.random_seed)
    #--model_output_file
    model_output = load_model_output(args)
    #frames is from problems.json, frame_ids is from pid_splits.json
    frames, frame_ids = load_data(args)
    # frames, frame_ids = load_data_gpt(args)
    result_file_name = get_result_file(args)
    #enumerate the 1st dict: image and ids
    for i, frame_id in enumerate(frame_ids):
        frame = frames[frame_id]

        image = frame['image']
        qas = frame['qa']
        corrects = []

        model_frame_output = model_output[i]
        assert model_frame_output['id'] == frame_id
        #enumerate the 2st dict(qas): qestion, answer, task...
        for j, qa in enumerate(qas):
            if qa['task'] != 'closed choice':
                continue
            if qa['task'] == 'closed choice':
                question = qa['question']
                choices: List[str] = qa['choices']
                true_answer: int = qa['answer']

                # random_guess: int = random.randint(0, len(choices) - 1)
                completion = retrieve_completion(question, model_frame_output['conversations'])
                #completion_to_answer（answer from gpt，list of true answer）
                pred_answer = completion_to_answer(completion, choices)
                #If value of question[] not in results{}, new a value for acc_counter(), and initail value is 0
                #which means it should give a new question and counter form 0 if it is new
                if question not in results:
                    results[question] = acc_counter()
                #correct is var，corrects is an array，empty as default，add a value every loop
                correct = bool(pred_answer == true_answer)
                corrects.append(correct)
                #results is[][]dict，question_overall，frame_overall are both keys，total and correct are in acc_counter[]，
                # results[question] is the questions it show，value is times it show
                results[question]['total'] += 1
                results[question]['correct'] += int(correct)
                results['question_overall']['total'] += 1
                results['question_overall']['correct'] += int(correct)
            else:
                question = qa['question']
                true_answer: str = qa['answer']
                
                if question not in results:
                    results[question] = acc_counter()

                # random_guess: int = random.randint(0, len(choices) - 1)
                completion = retrieve_completion(question, model_frame_output['conversations'])
                # pred_answer = completion_to_answer(completion, choices)
        
                # breakpoint()
                
                results[question]['total'] += 1
                results[question]['correct'] += sentence_bleu([true_answer.split()], completion.split(), weights=(0.25, 0.25, 0.25, 0.25))
                # # replace None to empty str
                # true_answer = true_answer if true_answer is not None else ""
                # completion = completion if completion is not None else ""
                # # Then operate split() 
                # results[question]['correct'] += sentence_bleu([true_answer.split()], completion.split(), weights=(0.25, 0.25, 0.25, 0.25))

        results['frame_overall']['total'] += 1
        results['frame_overall']['correct'] += int(all(corrects))

    print('===== Results =====')
    acc_dict = compute_acc(results)
    print(json.dumps(acc_dict, indent=4, sort_keys=True))
    print(json.dumps(results, indent=4, sort_keys=True))
