import json
import os
from typing import Dict, List


def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json'), 'r'))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json'), 'r'))

    frame_ids = pid_splits[args.test_split]
    frame_ids = frame_ids[:args.test_number] if args.test_number > 0 else frame_ids
    print(f'Number of test frames: {len(frame_ids)}\n')

    return problems, frame_ids


def load_data_trainOnly(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json'), 'r'))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json'), 'r'))

    frame_ids = pid_splits[args.train_split]
    frame_ids = frame_ids[:args.train_number] if args.train_number > 0 else frame_ids
    print(f'Number of train frames: {len(frame_ids)}\n')

    return problems, frame_ids


def load_data_gpt(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json'), 'r'))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits_gpt.json'), 'r'))

    frame_ids = pid_splits[args.test_split]
    frame_ids = frame_ids[:args.test_number] if args.test_number > 0 else frame_ids
    print(f'Number of test frames: {len(frame_ids)}\n')

    return problems, frame_ids


def get_result_file(args):
    result_file = f"{args.data_root}/{args.exp_label}_seed_{args.random_seed}.json"
    return result_file


def acc_counter():
    return {
        'total': 0,
        'correct': 0
    }


def compute_acc(results: Dict[str, Dict[str, int]]):
    acc_dict = {}
    for metric in results:
        acc_dict[metric] = results[metric]['correct'] / results[metric]['total'] * 100
    return acc_dict


def load_model_output(args):
    model_output = json.load(open(args.model_output_file, 'r'))
    return model_output


def retrieve_completion(question: str, conversations: list):
    if question == 'How many lanes in current road?':
        question = 'How many lanes on the current road?'

    for i, conversation in enumerate(conversations):
        if question in conversation['value']:
            return conversations[i + 1]['value']


def completion_to_answer(completion: str, choices: List[str]):
    for i, choice in enumerate(choices):
        if choice.lower() in completion.lower():
            return i



[
    {
        "answer": "Normal city road.",
        "frame": "FR8378",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2861",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9956",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4008",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9854",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR868",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2195",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7402",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8721",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3025",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11439",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5163",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5764",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9060",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12001",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11964",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7244",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1794",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10201",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11788",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8409",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5351",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9565",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9902",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13172",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5555",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3398",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13248",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR528",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR606",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2539",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7002",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5943",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11286",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10698",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3779",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11965",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4436",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR509",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5696",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2147",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12209",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13648",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13349",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9204",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1550",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6289",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10811",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7958",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4723",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9524",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4087",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12434",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5466",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10958",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12551",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4589",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4210",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1953",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6179",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6966",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR938",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12790",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5948",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7378",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4784",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4939",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12092",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3043",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR463",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3298",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2270",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10246",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9542",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12004",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4664",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR418",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10262",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13502",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11646",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6495",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8845",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13079",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8336",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5875",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6791",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4004",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13147",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR592",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2949",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7415",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR534",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2971",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10521",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6248",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11767",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12995",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11296",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9836",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR431",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10299",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1214",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3086",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10061",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8570",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR493",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11003",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR558",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11622",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7206",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5075",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11960",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12603",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6838",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4901",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR52",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3496",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2258",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13175",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR12360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12529",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8124",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2046",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6048",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3072",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12854",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9822",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10896",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13021",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2529",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13084",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3617",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7084",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4756",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10039",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1447",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1566",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11580",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1554",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1298",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7790",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9873",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7225",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8910",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5848",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2126",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13651",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR449",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR944",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR688",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11041",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12602",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9003",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR696",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7987",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2473",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5683",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11697",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12082",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6274",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5018",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12948",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12739",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2964",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2367",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12634",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7920",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9473",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4098",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13302",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6302",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR5123",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3472",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4932",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3449",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1828",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1956",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3058",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12134",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2314",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8708",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12766",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12665",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8577",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10527",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR309",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1749",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7406",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7624",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1206",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8962",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9941",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7202",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10684",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6625",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13015",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR149",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR753",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12539",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3666",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12619",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1329",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2705",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3628",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR326",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8735",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2271",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5921",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2534",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13144",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9639",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10055",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR949",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8899",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13524",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Roundabout.",
        "frame": "FR2209",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7099",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4089",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11381",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4654",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6704",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6357",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12265",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4545",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5093",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13382",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12823",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4353",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6748",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7056",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12954",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3905",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3080",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4875",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6883",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4132",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4917",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8620",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12975",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7707",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5743",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10181",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3515",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1675",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11968",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9503",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11457",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11118",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12938",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2588",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1696",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12184",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR374",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9157",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9569",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7044",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR644",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10295",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2869",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11587",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8802",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8053",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10381",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4433",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2653",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9811",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2419",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12572",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12331",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12754",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6959",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR161",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11304",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9196",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8584",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10913",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5410",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13065",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7803",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12252",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6850",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13463",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9627",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7843",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6080",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12680",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR495",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6594",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5261",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10807",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6127",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13140",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5401",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2540",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1438",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1397",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3721",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1905",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4202",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5354",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10431",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5367",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1032",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11707",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13024",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8639",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10548",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13520",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3614",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9468",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1269",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4403",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10948",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2336",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8956",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13376",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12586",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13166",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6857",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9613",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4584",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5834",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3049",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7819",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6665",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1131",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8836",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5409",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6930",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12291",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7098",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4365",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7229",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13091",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11675",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12860",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2401",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10042",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4783",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12387",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5927",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11943",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1654",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6713",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8056",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR69",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1450",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6363",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6695",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR68",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10298",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7926",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2508",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7190",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6607",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7145",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10251",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2285",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2762",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12835",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5140",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12040",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10715",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2106",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3384",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1858",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12907",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9539",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10206",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9856",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10935",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7363",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11773",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6516",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1820",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2670",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8430",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3090",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5714",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7396",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3207",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12973",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12125",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2666",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1244",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8411",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6801",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4496",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9365",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8038",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13329",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5954",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8868",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5844",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8953",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5423",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8951",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3815",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6992",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7438",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1525",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7107",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12882",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4215",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4101",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5603",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5806",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11413",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4047",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7140",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10633",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12309",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR36",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7130",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7995",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5816",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7766",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11712",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6339",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3756",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10676",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2243",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3867",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7337",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2338",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR43",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2394",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10680",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3730",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2802",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8252",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR506",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4854",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7554",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10163",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9898",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11747",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR288",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9986",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7105",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11397",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5907",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR884",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13192",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR73",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5099",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12239",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11374",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1590",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12792",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8748",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9915",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3774",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7964",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1500",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2495",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11084",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7000",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10388",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4883",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7974",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4140",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7730",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4037",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13073",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2786",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2571",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10863",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2112",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1949",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1076",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR499",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7700",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9118",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2063",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2606",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6869",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR17",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3242",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7886",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11794",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5117",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR239",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7728",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5259",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9004",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9976",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7824",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13622",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11714",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12389",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12376",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3516",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5195",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11401",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3071",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR811",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13232",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5856",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11813",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8127",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR484",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3290",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7679",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6670",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12701",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5808",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11562",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5050",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13096",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5773",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR727",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8006",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13508",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5138",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3050",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10617",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6890",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR864",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13572",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR125",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10976",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5567",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11330",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2443",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR537",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR32",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11644",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR912",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3161",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4554",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3053",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR65",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4954",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11655",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13611",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2084",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11131",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10747",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11984",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1827",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8842",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13265",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11997",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9066",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11284",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6502",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7334",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7288",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7629",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12326",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2053",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2406",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10499",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1017",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12392",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13338",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3251",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6195",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2678",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8398",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9670",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10486",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR551",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR709",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11058",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR565",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8890",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2208",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12681",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9512",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2734",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12841",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13141",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR434",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6185",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1656",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8493",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR130",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6839",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7291",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3671",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10440",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4628",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4650",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10502",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR886",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11889",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7586",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13179",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2281",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5523",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR650",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1295",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR781",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4129",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8077",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13125",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9743",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1355",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1514",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5433",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7025",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11258",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7182",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9686",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12179",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR940",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2251",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13587",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12341",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1797",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11719",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4581",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5803",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2944",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8763",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6955",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8918",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6233",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8506",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR616",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11034",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9591",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7160",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13490",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR552",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7788",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2838",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12896",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4034",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3840",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5494",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5663",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2200",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR656",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10252",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10925",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11467",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2076",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7588",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5918",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8503",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10837",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4322",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4764",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11373",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11664",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3588",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10349",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8048",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11510",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7917",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12526",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11299",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10551",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5742",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8400",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4074",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6006",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5718",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2238",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11642",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9129",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4996",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11225",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1013",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8097",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5465",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12003",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1342",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7821",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7698",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8780",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12061",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7409",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6123",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2130",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4247",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12141",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7622",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6226",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8296",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2229",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10946",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5266",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10887",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8422",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3274",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11259",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR185",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1271",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2645",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9470",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12923",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12832",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1559",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8943",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9442",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7919",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13574",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6651",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8302",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12566",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6033",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1484",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7496",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5785",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9495",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7285",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR220",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7203",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9820",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11578",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2306",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1358",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4463",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4018",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10498",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1110",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7163",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7068",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4591",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2955",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10475",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5393",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11763",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13713",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11881",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8379",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1470",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10606",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13623",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7666",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5645",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5973",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7485",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4988",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8423",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6956",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7587",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10964",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10112",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9082",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6217",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3799",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7944",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7861",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3063",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3075",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4594",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1843",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12904",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5713",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9750",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2541",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10411",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12042",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9929",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4050",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12210",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2352",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1239",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8237",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5054",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4649",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9631",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12958",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6132",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11369",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5286",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9951",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR368",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1270",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5417",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2496",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11082",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5814",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10787",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10795",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1994",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3636",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13450",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1756",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6338",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13095",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12800",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9065",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR876",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2806",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR314",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7021",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13440",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5906",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11044",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13123",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6767",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1606",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7576",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12588",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2875",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7063",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11090",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8020",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR642",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13744",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8828",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12046",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3974",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3858",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3847",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7933",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6142",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR467",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9600",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3132",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2843",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6078",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1959",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6168",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7272",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8680",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7687",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4773",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7490",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11590",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10653",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13384",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3964",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13240",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3067",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5592",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3369",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10117",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1412",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2860",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12446",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1842",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6082",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5127",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3816",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11052",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR681",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9397",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6282",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5800",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11855",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4575",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12412",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6881",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3913",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11465",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13649",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1020",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5712",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8528",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11233",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12629",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3370",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1093",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2851",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2497",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9614",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11163",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3510",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7518",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2664",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10229",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10497",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2033",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9688",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13596",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7787",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3909",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8657",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10574",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2398",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8387",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR951",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11358",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6402",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12266",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12088",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4182",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR929",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8752",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11824",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13206",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13434",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10572",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3491",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12687",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4073",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6151",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10323",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3112",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1962",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5281",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6494",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8163",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5215",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1416",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13620",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1433",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10302",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7506",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR560",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7523",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10756",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7352",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3631",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5239",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8975",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3416",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10194",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11102",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1563",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12942",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1218",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8870",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5958",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3685",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9763",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1092",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4531",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5035",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR666",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7709",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13076",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4621",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4943",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6075",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7410",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12460",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9739",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10342",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5198",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6301",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4715",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13416",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6015",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5955",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9230",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2620",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12258",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3068",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9038",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12401",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9806",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8238",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10446",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10449",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10841",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7831",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7664",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12768",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3288",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8926",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3434",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3146",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8524",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9370",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9711",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6932",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6085",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR780",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR280",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5870",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13367",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2776",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8201",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12459",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5902",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3500",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6821",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5041",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4607",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8264",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4804",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7567",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9057",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7484",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11715",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9227",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8738",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2202",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5577",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11676",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9560",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR133",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6758",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10118",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7383",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11585",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4306",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1684",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3022",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5078",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3918",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10057",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6325",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9033",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9837",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5619",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10142",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9385",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7172",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11647",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6523",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7896",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13311",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8608",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3297",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6776",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR102",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10780",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9311",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13187",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11665",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13016",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10187",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5997",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13388",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8092",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13301",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4913",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1707",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3466",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4358",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6994",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6847",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10507",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5443",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2864",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2061",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5151",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8730",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR826",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7070",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9347",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3623",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9498",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11216",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR655",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11840",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2546",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR844",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8944",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6898",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2062",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13549",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9690",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1589",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6202",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3508",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10355",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1370",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10810",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4973",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3020",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR152",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5908",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3754",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13408",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8102",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8383",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10578",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6465",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9553",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12607",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4393",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6755",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8505",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3266",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10700",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1801",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4758",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3613",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8221",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2015",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13400",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8230",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11658",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2341",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10202",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2478",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1574",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13120",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8435",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9075",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7934",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4562",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9222",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3566",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8893",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9160",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1701",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5119",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11695",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2549",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7880",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6934",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7568",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7359",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8446",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9190",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2485",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1516",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8001",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12250",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3124",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3338",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5920",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3394",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4269",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4230",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR289",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7578",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8431",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8883",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7548",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR163",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11175",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3041",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4727",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11024",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10020",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9329",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7998",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4285",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9113",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11035",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11072",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13760",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6855",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10770",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6187",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4170",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10663",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9798",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12323",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13198",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5956",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR800",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12992",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7357",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7210",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1327",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4701",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1290",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9342",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4165",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12831",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9663",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10393",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5827",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR27",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2683",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2403",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9853",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5531",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7147",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8034",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6050",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1714",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4419",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR651",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3222",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10065",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8968",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9295",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1633",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7932",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4003",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7459",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR708",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2265",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3355",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11513",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12910",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11556",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6769",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13667",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2628",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2973",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7701",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2755",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2932",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4861",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12707",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2402",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1384",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13070",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8332",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3367",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8531",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6808",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR775",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9712",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2778",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2029",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2134",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6059",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4469",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8895",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8368",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9483",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12147",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13379",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2441",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5363",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5049",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR601",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1201",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5478",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8737",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4225",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6827",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5789",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11301",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8113",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1068",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12314",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR749",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3185",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5449",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9963",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5015",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2523",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12491",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2142",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8217",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8007",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR787",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12877",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12932",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10822",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4256",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6515",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8816",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13445",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10157",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13605",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6406",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4720",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8790",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4570",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5418",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12788",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2792",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11169",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6963",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10845",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7050",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12537",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11049",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12013",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11500",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12764",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1826",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9067",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12552",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3544",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11520",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7010",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2472",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4017",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13601",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4878",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4499",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7769",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6207",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6349",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR44",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1926",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5563",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2727",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13627",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10686",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9655",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10438",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7972",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13135",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7686",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3804",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5632",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11095",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6381",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4802",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9887",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8623",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8695",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13309",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9658",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9313",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5914",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1451",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6157",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11215",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1010",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7621",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9687",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12410",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5463",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8215",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5020",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6654",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11656",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13678",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2172",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2574",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11441",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3889",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10760",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5868",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR188",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR59",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6196",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6461",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4624",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12024",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2087",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9381",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11809",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4940",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8546",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8621",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1904",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7585",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2175",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11333",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR265",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9959",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7828",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10725",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3839",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6595",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7571",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6684",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11514",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11484",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9224",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR399",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR433",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6508",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5166",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3475",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3970",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9451",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8078",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12133",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6622",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11972",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12233",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12987",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2873",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7913",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5515",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4819",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7890",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9841",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4413",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4022",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR979",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1065",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5705",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12478",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6576",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13046",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3218",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR612",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9087",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5805",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4502",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5357",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6893",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7322",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5390",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5512",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12733",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2721",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11576",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR931",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7393",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3795",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11339",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11854",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10077",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2298",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7382",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5737",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3195",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11313",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR96",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3070",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR468",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6062",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2021",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4897",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12527",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2426",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR456",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR718",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR881",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12116",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9093",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11979",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4915",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5824",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7589",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8685",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8796",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2276",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1693",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7626",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6397",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4080",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11408",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1102",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6385",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11632",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12475",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13060",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12362",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7422",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR896",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12647",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR556",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12581",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9200",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1722",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8527",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2091",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10709",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7509",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7755",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3157",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10794",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11119",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR49",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2311",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3524",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13101",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6586",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3720",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5574",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10541",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1853",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6851",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9165",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2221",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7354",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8624",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7223",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11083",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6206",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2425",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9997",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1085",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13589",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR214",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2845",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3877",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12373",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6300",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12738",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8304",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4847",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7227",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3819",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR937",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2923",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8875",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3675",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR594",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12382",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8739",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8668",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5912",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR80",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3941",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8496",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11194",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7785",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4384",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8519",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1758",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7419",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8784",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10941",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5440",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR711",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12622",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11893",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9013",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8820",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR497",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4969",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2780",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5557",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9787",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3700",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2513",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3015",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6037",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8195",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3002",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3246",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5525",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13731",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR19",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5698",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9809",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1176",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4950",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR221",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3001",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1488",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6743",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11939",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10158",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR544",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4062",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10409",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR995",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3017",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10284",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7138",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7555",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4332",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7781",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11417",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7521",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2842",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3827",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2030",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8408",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4508",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7619",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3806",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13465",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7090",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1332",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9177",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6436",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11797",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11334",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5214",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8440",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5886",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4616",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR962",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1223",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6022",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5758",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11529",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR853",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR18",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR352",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3940",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6011",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9790",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4414",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6309",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7335",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR151",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9411",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8455",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3362",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11698",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12670",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9889",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2844",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8285",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6241",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR22",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2140",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2048",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10901",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6510",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11806",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10960",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7219",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3062",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3642",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2962",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7592",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5770",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9979",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11973",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11061",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3506",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6940",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12454",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4989",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR610",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8002",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9894",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1543",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4977",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8553",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3327",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11962",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9043",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8762",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5913",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9552",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3208",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13693",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10341",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13247",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5289",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10911",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3417",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5005",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9001",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1434",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4038",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12631",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4894",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR21",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9314",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13132",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9243",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2365",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3701",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR105",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1337",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6070",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4613",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5847",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2451",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2632",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4327",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8754",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6472",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3577",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12222",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2004",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6653",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9633",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3864",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5715",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10932",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7418",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5217",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6913",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2237",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR600",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5232",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7165",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4541",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7516",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9030",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13562",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2572",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13081",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2301",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3450",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11046",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13067",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11394",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6888",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2486",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5011",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4015",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13292",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7016",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6752",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10690",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4135",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6327",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13236",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11447",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6117",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7768",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1259",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5042",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8392",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6262",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6518",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2067",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8280",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3552",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5616",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2332",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10417",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5863",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13556",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2989",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9985",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR939",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5392",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9775",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3512",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6177",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2646",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11389",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4357",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR803",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR78",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2515",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3616",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4631",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10796",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8817",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8652",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13679",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10816",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6101",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR947",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6681",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11474",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5901",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4629",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9141",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12544",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13161",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2291",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8659",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR4683",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11946",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3176",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5147",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR907",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9918",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11497",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3870",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13264",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2604",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11300",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5439",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11548",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5990",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR752",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4774",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6923",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8376",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2927",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8262",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6696",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5846",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2850",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9296",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10160",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6652",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4907",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5580",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5295",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3635",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10097",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR38",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7905",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR869",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10238",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6686",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3359",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10671",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR266",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12509",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1023",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5756",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11525",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9494",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8961",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12888",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10764",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9216",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5750",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2240",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7102",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR863",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2269",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8218",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8126",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10401",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6332",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12017",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5797",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7956",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11268",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12931",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12897",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3797",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5878",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3389",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8809",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9803",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10516",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR282",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10367",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3314",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6176",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11735",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3169",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8999",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12285",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12370",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1899",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6025",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6840",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10802",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4840",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR118",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9576",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1893",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7132",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6194",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4349",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4578",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8650",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5194",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13390",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8276",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6521",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5024",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2188",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5722",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4944",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6116",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4450",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5193",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7194",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7595",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3632",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2626",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7201",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3663",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1253",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9892",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4297",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6198",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8467",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8564",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3705",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8666",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5717",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8894",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7149",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3048",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4166",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3558",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3778",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1066",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11116",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9575",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10605",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6507",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6643",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2834",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11074",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8896",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3170",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7612",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8200",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12256",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11519",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4747",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR95",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR12294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9846",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4794",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6384",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8010",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5895",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR548",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1007",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4461",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7483",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11198",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10493",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2890",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11814",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4239",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1166",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6374",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8250",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11161",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR385",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR627",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR511",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR714",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1930",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12895",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR232",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5244",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR959",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5044",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4787",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2651",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7062",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1804",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1998",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR626",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6779",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10813",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3927",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5684",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10550",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7207",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6493",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1000",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5219",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7442",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2135",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR83",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9731",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13633",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3361",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5105",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6979",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12282",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9861",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11906",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2347",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9734",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9307",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13337",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1592",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11415",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13027",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10883",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3911",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4619",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5258",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4338",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2219",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12117",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7295",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3483",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8607",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11499",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4388",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9000",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3311",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2938",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3708",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8787",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12774",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2418",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1224",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10977",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7904",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4651",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3130",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4240",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2177",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4526",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13099",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10897",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11818",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR738",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8359",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10569",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12628",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13377",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10900",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4224",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8133",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2260",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1989",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12842",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11241",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6640",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3387",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7289",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3766",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR809",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13260",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2958",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9848",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8066",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6880",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7648",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3436",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11862",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3322",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7126",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10805",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10179",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12394",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13092",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9112",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4893",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2952",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4805",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12181",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6242",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5994",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13209",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8080",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4839",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1511",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7887",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10131",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6231",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10636",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10844",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1117",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4908",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6013",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3759",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR422",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11123",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9241",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9907",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7414",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2527",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4401",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11081",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7734",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR26",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7753",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2950",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11402",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8313",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11378",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7220",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4115",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7799",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3220",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5817",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9288",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12249",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12765",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3107",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4281",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3638",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6549",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5483",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2374",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1095",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6719",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3498",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11277",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6915",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5161",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5196",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7583",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2262",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13169",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9424",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6400",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1499",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1765",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9607",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8068",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8123",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12149",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10717",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11978",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11582",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4488",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4356",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR913",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR690",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7451",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10881",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2410",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9547",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1784",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5396",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11092",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR168",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR99",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13385",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2412",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5560",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1747",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2583",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4458",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3354",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6130",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7962",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7164",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12010",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9279",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7884",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12789",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR686",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9882",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12486",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR646",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4195",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10952",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5688",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11468",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6931",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13491",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3547",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5511",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3801",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13600",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3221",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12817",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR986",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5415",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11690",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12077",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6444",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13336",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3644",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1025",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2856",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8728",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1323",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10798",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9781",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5235",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10135",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4470",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3551",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10940",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9721",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7274",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13645",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12805",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9756",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3996",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1351",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11247",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9975",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1164",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1177",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3018",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4331",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3267",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6926",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3188",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11636",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2708",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1583",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2826",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7883",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10790",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR192",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9773",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9881",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3382",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4987",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8005",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10850",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11910",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8747",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR989",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7909",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5565",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5787",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8003",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3805",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5809",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5115",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8966",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1939",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9543",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5404",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2693",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10622",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7425",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11594",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1965",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7569",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5023",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6690",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1700",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Roundabout.",
        "frame": "FR4396",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2607",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7029",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4434",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8062",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5754",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5058",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10520",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8712",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7606",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5665",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5534",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12435",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7915",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6884",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4495",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8994",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9606",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6404",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9818",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8309",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8626",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10814",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6153",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8181",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2022",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5516",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6862",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8141",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7649",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3875",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9519",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1309",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1330",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8187",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR861",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2617",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8268",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11816",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3582",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13707",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10281",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1966",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR1558",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3935",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10789",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1086",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12506",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12696",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6090",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5485",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5144",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3845",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7841",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7144",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1643",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13668",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10515",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3699",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6513",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2526",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2413",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10739",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6631",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7544",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13261",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11149",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6330",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR933",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8085",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8852",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3834",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12263",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11557",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8818",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8671",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13466",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1671",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8397",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13249",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1704",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6370",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6446",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3680",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1486",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR412",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12243",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9127",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8508",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8727",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6941",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9529",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR10577",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR252",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1727",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6546",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6096",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1864",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2537",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7613",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12864",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12650",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4491",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5284",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6920",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR182",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5047",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13370",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3059",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11352",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11452",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9950",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2559",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9077",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11948",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9208",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7866",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2986",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6600",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1968",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2373",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1024",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7348",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR203",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13503",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9728",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3923",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6929",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6247",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8902",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11657",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8955",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5845",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8156",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4045",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13588",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13433",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7704",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3478",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2558",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2713",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1424",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8929",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8646",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1878",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10068",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5430",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2378",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6045",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10539",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9485",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8719",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7038",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10943",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4030",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4686",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7750",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR237",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2862",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5934",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5230",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR603",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3138",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13138",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR339",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7608",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6984",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8055",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10274",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10693",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1532",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10263",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8175",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7397",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12054",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8248",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR621",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR193",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11382",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8325",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10659",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6820",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6506",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11670",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8619",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5858",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10997",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5203",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12821",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13072",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR491",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR632",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12986",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR57",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8396",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5384",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR792",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13577",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10625",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8125",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3399",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9280",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9893",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR116",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1260",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7266",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2521",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10788",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4001",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10496",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2460",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9116",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7306",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12217",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7721",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11859",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4778",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6426",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9187",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12964",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12645",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9795",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10610",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2450",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10990",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7923",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9888",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9742",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12644",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11515",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3933",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4975",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9071",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11011",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2722",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10839",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9659",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9748",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8552",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11164",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13753",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7449",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8415",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3984",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8122",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2259",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6218",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10553",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11584",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10738",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13498",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4006",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR488",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7269",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1759",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7175",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5381",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6413",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6401",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR10691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3426",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3392",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8678",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12911",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8039",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4333",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6486",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5591",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7605",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13238",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12023",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11894",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6566",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2483",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12657",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1373",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11629",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11988",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9229",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6504",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10806",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10817",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8649",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5631",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3184",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8559",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10060",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5277",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10044",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7092",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13185",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4168",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8161",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9284",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11564",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10909",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8976",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5253",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3282",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6830",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5636",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12203",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13105",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1446",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2886",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11449",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11015",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10873",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9701",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13259",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2334",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8021",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6899",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5389",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR167",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4748",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8329",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5475",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8827",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7209",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2070",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2103",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12651",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3643",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2775",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2125",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1602",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10511",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2235",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11975",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4734",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7949",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12786",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11623",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13298",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1147",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4314",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7784",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11252",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9770",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10848",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11971",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11129",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4242",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2915",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7695",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3828",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13312",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR706",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13606",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10418",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR308",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7146",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5394",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5046",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9635",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11059",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1041",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12115",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11951",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1951",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10207",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1723",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR742",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5601",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2214",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11844",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1403",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11178",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8142",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3529",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12000",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2013",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1067",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13093",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3958",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6701",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11251",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1887",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4369",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13033",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3711",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6822",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12743",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5932",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12403",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9532",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11355",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11188",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7961",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3006",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3600",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7325",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11162",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8881",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12730",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13369",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12649",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR123",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4984",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3994",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7897",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13013",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4145",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR755",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11050",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR835",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11429",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7553",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9577",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7020",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1607",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6443",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR455",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13034",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4540",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10582",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3081",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7204",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4995",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7705",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6826",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3712",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7733",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1668",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12493",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12708",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6583",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1955",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6032",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7188",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1483",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5678",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10584",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR58",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4179",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7640",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4653",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5988",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11206",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4549",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2996",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13513",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6475",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3497",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10456",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9062",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11410",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6548",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1732",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10296",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8216",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3648",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12276",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6744",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10266",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2068",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9221",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10616",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8988",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3425",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11541",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1415",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11553",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4192",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8722",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7181",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12933",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR819",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4082",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11488",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2339",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5620",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8030",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5910",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8058",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1363",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR778",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7231",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3520",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10552",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4951",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9460",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12163",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12016",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7101",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11961",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8979",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13664",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11645",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4095",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9032",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13739",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6687",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8117",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9650",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8447",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9358",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1289",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR769",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3052",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5503",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9413",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10430",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9041",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4235",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9354",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12322",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10461",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11271",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12033",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10708",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2758",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3344",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8676",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5528",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2445",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4042",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2912",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10544",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10565",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8146",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13418",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1263",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13438",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2503",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR40",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10333",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12565",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8138",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13291",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2820",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4780",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8778",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8060",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1808",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13411",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR998",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13635",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12940",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8541",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6894",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8416",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6105",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR342",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6040",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10929",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1734",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR889",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10870",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8249",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5640",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5732",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10064",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1876",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2158",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12027",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12858",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9232",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12589",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5837",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12049",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR549",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6540",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4286",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5695",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10687",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12878",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7525",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2001",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8052",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11635",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8232",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9022",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10818",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3739",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2198",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13559",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2832",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5520",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5791",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR568",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12828",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8041",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10166",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5558",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10635",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7252",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11558",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11354",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1917",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3585",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12845",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9934",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5473",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13304",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4866",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7947",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10090",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2948",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2446",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10791",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR169",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8110",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7271",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8643",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8800",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7617",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9744",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7162",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3665",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11179",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1615",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8704",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1172",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6572",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11107",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11416",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10004",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3906",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13134",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR410",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12513",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7985",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10071",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6382",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR12060",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5280",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10119",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6511",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3857",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12085",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8887",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13592",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10133",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7663",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7601",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5225",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6749",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4055",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6268",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10132",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11915",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1469",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2141",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR473",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11426",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11407",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12837",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7054",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12505",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4505",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3505",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10370",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12472",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9349",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13242",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12810",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13184",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12870",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4243",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12803",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10537",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5959",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4731",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4402",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7941",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7800",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8638",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10063",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4576",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13747",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11781",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11950",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10049",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7528",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12200",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6378",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR242",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10218",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4838",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6699",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7208",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10890",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9438",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1988",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11281",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12467",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12029",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10519",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8244",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9018",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12202",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1831",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8614",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7989",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11540",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9541",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7859",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR454",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1973",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11619",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9084",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7429",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1410",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13058",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9554",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11974",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8441",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5160",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5830",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4976",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11383",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7344",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3456",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6837",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8333",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6044",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12465",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10280",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR748",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR430",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8209",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6055",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4602",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5998",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4300",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3895",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4797",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11436",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1510",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4063",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3447",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13737",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1423",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11195",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4955",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11994",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11873",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9360",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3952",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6974",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4335",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5854",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1779",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2078",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10549",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12299",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10441",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7732",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6368",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8346",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9953",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7097",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7627",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5725",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8099",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7127",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1534",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3271",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3892",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11470",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9175",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10797",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12769",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12135",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12073",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10447",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5568",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3836",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2639",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10776",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13063",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12146",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5755",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1081",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13735",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5532",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR347",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5649",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3460",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12057",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3658",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12518",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6527",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1910",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6238",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5292",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8369",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR860",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9548",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6953",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5000",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3811",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4430",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5829",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11298",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11621",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8847",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4373",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9685",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8288",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13126",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6420",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR324",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8227",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9578",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10634",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10027",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11214",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3880",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR273",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10576",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2326",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5793",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8540",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1639",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9337",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2337",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1608",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8545",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2582",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4466",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10742",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3887",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5925",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11605",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9905",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4011",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12677",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10801",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4786",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1844",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6503",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR527",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11459",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11221",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2476",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2323",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13085",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR731",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13071",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12914",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10678",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR388",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4919",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5581",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8277",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2074",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10972",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11246",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1038",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7718",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1810",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2779",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9515",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11412",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4931",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12944",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6026",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2635",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1371",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10422",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2732",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6554",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6222",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1690",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR285",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7027",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR953",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7770",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1883",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5986",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5953",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11896",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1231",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10413",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12965",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3096",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5293",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11007",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3390",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7671",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9449",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8901",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9439",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5762",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR67",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11070",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2999",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7860",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6786",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7789",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5521",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR251",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7242",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11311",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6113",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10237",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12929",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11160",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5670",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6199",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4959",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11843",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1582",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12883",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8247",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1429",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7253",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9968",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8835",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6768",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6073",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12058",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9211",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2184",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4237",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR968",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5681",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13098",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4156",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3201",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11038",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Roundabout.",
        "frame": "FR5087",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10292",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4779",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10970",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11680",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3110",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4290",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2313",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8382",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9895",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4452",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5027",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5704",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9549",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8225",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8930",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7048",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13224",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13271",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8331",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2264",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR987",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12796",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9766",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6367",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9392",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR396",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR424",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5180",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12884",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9505",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4656",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR963",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9733",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3629",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10426",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4445",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10122",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1296",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12101",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2884",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR996",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3820",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8456",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR900",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5103",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4133",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13454",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8374",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3921",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR595",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6047",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9454",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12990",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9693",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4755",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10170",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11808",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13475",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8158",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR34",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9276",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5783",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12433",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2101",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5608",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6650",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8952",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10242",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2548",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10596",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1678",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4680",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3099",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3419",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7616",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8514",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8162",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2138",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8808",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13237",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4498",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1155",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6659",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7823",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10007",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2562",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11821",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR898",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7284",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7457",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11133",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7681",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2274",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12950",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13268",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9819",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4334",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2049",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5398",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11185",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12438",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8115",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3534",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR582",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9585",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4942",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12778",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13542",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13473",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2764",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6419",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4309",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6221",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4864",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3963",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3519",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9090",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12570",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4666",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12347",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3937",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3873",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5125",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11617",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10182",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4208",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13113",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3441",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5999",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7562",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13758",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11732",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7330",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6377",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3794",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3624",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1709",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12827",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR107",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR478",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4781",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3435",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9910",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8491",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7636",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR914",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6707",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9623",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9419",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1162",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR821",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6613",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6648",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13181",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10002",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12583",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10995",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9780",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7623",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4241",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2092",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3233",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7829",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11944",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3647",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11329",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR984",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3039",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8022",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11217",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8035",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6784",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8571",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1644",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12653",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9267",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2122",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13258",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR207",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8872",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8603",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5798",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1436",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3127",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8460",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11914",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7881",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9374",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR507",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12557",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5744",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2247",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12262",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5331",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11783",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5210",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1051",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3791",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7177",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6240",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5595",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6046",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7572",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9589",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13643",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4134",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7124",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5224",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12905",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6346",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8909",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13767",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12962",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12458",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12971",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10728",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR570",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6126",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR483",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11830",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10110",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3652",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1896",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5995",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12043",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2652",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6732",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5564",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6488",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7428",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10433",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10888",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12542",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13116",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13325",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4882",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11424",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11409",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7263",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10991",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11546",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9378",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6804",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13425",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9414",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2591",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4752",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6007",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8518",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9333",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11614",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12727",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11789",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2392",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3838",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1042",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9388",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8170",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8711",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4739",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7786",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR425",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5929",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10735",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7018",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR540",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2567",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9285",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3408",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11810",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1215",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR801",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8134",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12469",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR967",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR176",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5753",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9453",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7765",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9171",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11912",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6192",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4272",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11628",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9714",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3054",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7189",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7620",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3057",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7542",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7150",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8964",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8628",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7248",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9484",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10846",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3004",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2393",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11575",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6264",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12065",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11891",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12970",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2000",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8785",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1626",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7801",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4704",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR505",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9684",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4148",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13381",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12334",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9540",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11048",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3231",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11466",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5438",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7456",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8860",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR838",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR361",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR24",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1809",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12474",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4934",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1077",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8795",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5974",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8417",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10971",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10264",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4490",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9020",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4279",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12002",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12264",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6329",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR270",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1840",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6906",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13423",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1399",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1435",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13743",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5575",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5629",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12847",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11768",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2638",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6208",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13351",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4347",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7927",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR86",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5082",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5061",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR935",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12760",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12279",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12381",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8592",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12807",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11053",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10961",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8997",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5559",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12780",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10445",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7011",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8259",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR746",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5839",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1457",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1892",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7783",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2902",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6246",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8246",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11305",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10612",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12978",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5296",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3279",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4025",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11361",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR249",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11858",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4144",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6425",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3535",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9321",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13541",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11444",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12683",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR904",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10821",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8793",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10730",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2431",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3653",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4146",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3159",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7096",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7763",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1861",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR298",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6219",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2284",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5153",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1539",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1996",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8235",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11653",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1849",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10465",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR196",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1207",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10272",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1261",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1044",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6843",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7024",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9096",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR502",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3789",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1196",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR798",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8878",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8989",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12826",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12880",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8042",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4558",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5462",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5220",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6987",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4440",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12939",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6814",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12762",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2150",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7112",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11760",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4902",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9119",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10835",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6299",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3121",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10638",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9747",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12094",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9188",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3954",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4537",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7032",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3610",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1551",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13088",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11267",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11486",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9406",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11243",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1768",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR762",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4749",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1276",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6010",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5505",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8595",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9912",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR713",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7082",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11507",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4873",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4553",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6083",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10183",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1226",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4806",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9169",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8939",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5149",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10414",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3325",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12611",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9522",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1235",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR640",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3767",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11863",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5218",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10099",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8086",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4560",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10373",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6393",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12545",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4504",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3634",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1796",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2699",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3625",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8402",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9652",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3590",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1972",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9355",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8443",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1974",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8330",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1813",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5209",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8581",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10313",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9960",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR901",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7976",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1875",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5810",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13368",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7714",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1348",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR466",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7852",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5983",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8220",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3262",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4946",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10289",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12688",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12519",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2042",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8588",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7773",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2328",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11106",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11899",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR808",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9292",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3608",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6770",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4909",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5374",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6605",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1775",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7737",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13219",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13335",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2880",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3376",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8468",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR348",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5450",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8190",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11666",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13353",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12818",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1267",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8567",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7938",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1613",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6795",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7280",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5775",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13628",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1099",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7061",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6459",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12193",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10319",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6734",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1404",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6464",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9444",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9797",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12934",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9259",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12355",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4785",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5424",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2566",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7811",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3216",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9961",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4603",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2741",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR892",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR812",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13663",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR862",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10473",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10650",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2853",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10351",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4692",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10734",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9135",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8839",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8550",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9238",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3027",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4107",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9098",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5063",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8429",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5765",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7951",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7830",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12851",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR137",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7331",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2507",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5984",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6160",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4293",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5493",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4176",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11917",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9667",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12286",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8351",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4744",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4632",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11831",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4060",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8749",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13158",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4859",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12158",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4191",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7316",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6912",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5905",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10556",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5081",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8363",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR618",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6596",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11182",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4299",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9425",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4936",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11127",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5669",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR322",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1752",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9660",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5076",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8206",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7856",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4938",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7412",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6534",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11986",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5338",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR344",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4355",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9431",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6263",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8713",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5981",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR300",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9045",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12105",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1850",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2129",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4454",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7091",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3977",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11502",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7197",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR354",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7760",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5518",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7391",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5915",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6244",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5471",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2614",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6792",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13346",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7013",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6060",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11668",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6794",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13106",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR211",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11927",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13002",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8612",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2878",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7069",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR885",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6392",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1241",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7791",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR805",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5071",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8081",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1426",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10979",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11875",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1870",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR450",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12168",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4131",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11527",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11384",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12076",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR172",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR729",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2220",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8693",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR13294",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8938",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11207",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12574",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12711",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4141",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8315",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5622",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11362",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12859",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12038",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10459",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6313",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR605",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12695",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3192",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10575",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3458",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11405",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10869",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3664",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5855",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6161",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10762",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1771",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7844",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12977",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5822",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11626",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13435",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR585",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5348",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2448",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13690",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2627",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6537",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR2375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10076",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11577",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9927",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR9339",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR181",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9006",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9496",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2434",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12857",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11477",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6593",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6999",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6698",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13441",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8231",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7226",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10927",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13585",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4789",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4036",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7757",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8492",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13403",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6512",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6352",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1627",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5972",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2612",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12190",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7826",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3066",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4762",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7545",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2825",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10269",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6802",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4646",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6671",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3241",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3034",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12241",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4586",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1918",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10656",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR335",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6700",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9656",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9570",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13131",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3000",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5526",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1443",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10106",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1129",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12485",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR720",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11359",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2440",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6158",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2805",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10969",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8488",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR394",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1368",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10906",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4842",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7270",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2011",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12735",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4516",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6614",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9396",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8573",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12408",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5874",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5869",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4248",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3931",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2893",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12495",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13030",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3495",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR129",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12985",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR284",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR293",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8971",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR264",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8547",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8720",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6188",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR62",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11184",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10568",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5554",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12767",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR337",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR498",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5819",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11254",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2745",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12862",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9286",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8814",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4049",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6068",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10595",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR200",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7647",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5892",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2545",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10713",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7738",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7546",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4923",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9463",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10624",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13110",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6997",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2575",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1587",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9793",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11437",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5751",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2532",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4639",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9251",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12430",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12600",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10601",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2227",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2892",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3330",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6737",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6939",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12925",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2791",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9783",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6345",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4728",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3697",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10954",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5795",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5088",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2468",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12080",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11721",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8851",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9404",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4083",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3583",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1385",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9937",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12377",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3480",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5989",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2131",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12752",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1291",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4904",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1619",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4735",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6785",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9282",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9835",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5802",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR10109",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7342",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10116",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7779",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12758",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR5072",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9696",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5039",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3640",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3560",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13728",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12672",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12582",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7017",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR733",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11253",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12494",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12235",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR1506",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6662",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR843",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR132",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3378",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12086",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9327",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11687",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11192",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10103",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4143",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7115",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3469",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9281",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13276",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5479",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10108",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11609",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3602",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2421",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2303",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9715",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4718",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1645",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4053",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10443",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10963",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10828",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8478",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR680",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8566",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9334",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7761",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12523",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13639",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8826",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12579",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7834",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4489",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4173",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7467",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10775",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR442",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11332",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2031",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR8144",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11353",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9998",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7984",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3605",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2631",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12404",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR5397",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6150",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8301",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR3765",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3217",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11876",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6591",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1829",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5643",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11985",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10126",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7052",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7307",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7001",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10396",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7654",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1807",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12300",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6277",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7580",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8160",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3645",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR726",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11718",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13485",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5249",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR973",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4236",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5113",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11659",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10978",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12439",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7847",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR45",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6415",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8293",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10727",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3332",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5040",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5250",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7465",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4263",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR6509",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13730",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2840",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8823",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12488",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10871",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12808",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13210",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10282",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5066",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4687",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8733",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12865",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13287",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1359",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4898",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1710",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3365",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4777",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4543",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2925",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3190",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10490",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR783",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11472",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13318",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4507",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3773",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7114",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10564",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12358",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5707",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8145",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7599",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6524",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11956",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11395",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13412",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1156",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5799",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13295",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6387",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6759",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Road mark repainting.",
        "frame": "FR6145",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2743",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9598",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1720",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5472",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12342",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3100",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10994",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8281",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1408",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5102",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11272",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Roundabout.",
        "frame": "FR8323",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6624",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9376",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR77",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11533",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9322",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2553",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1165",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10518",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12095",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1688",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9461",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13557",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3117",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7258",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10587",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7151",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7482",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR357",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1526",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1283",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6253",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR647",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9709",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2911",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13636",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7878",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR12139",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3414",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9445",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13083",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2937",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12340",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR695",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8702",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12699",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2510",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4040",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13330",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3571",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11536",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5301",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4265",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6027",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10130",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2190",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12176",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11679",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4636",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1766",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2910",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5843",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR924",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2827",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10697",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9925",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9061",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8509",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5487",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5172",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10563",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12667",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12445",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3966",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR12074",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7458",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5930",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR764",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7774",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4869",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8892",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4876",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4652",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10524",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8019",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5985",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12963",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2584",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10903",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4698",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2050",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2400",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4376",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2056",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR48",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11376",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9972",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12415",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9879",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10627",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11151",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6938",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5909",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6278",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4968",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13404",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9047",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6091",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8299",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6411",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3131",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10985",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13145",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5922",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9525",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9218",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2772",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2718",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1056",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2947",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9852",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR35",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9980",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7129",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5457",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1724",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8682",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10920",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11711",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13150",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7898",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6136",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3292",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR519",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13342",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9833",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13074",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6257",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10936",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12694",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5904",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6646",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3859",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8427",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1070",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13102",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4947",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR228",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11835",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6009",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3796",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13657",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1594",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR3335",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12957",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4212",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9076",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13567",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4307",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4366",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6250",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10006",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR2320",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5952",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11610",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7692",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4900",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR8380",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1907",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6186",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5569",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13532",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR6458",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2452",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11021",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8442",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12721",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10219",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6538",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12661",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10968",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11921",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2837",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12028",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7853",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13205",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6914",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3641",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5760",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12091",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7468",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3253",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3383",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7691",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR952",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9850",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR736",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR767",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR4948",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13043",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR223",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR51",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3409",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4174",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1716",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2012",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8373",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1674",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10561",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7916",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8033",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2408",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1845",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9597",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11673",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3832",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1466",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10884",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR7156",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5335",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11337",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5059",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8982",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1387",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7604",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11177",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6255",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13363",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR523",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12113",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7675",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3424",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9046",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3786",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8353",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR9928",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4111",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7478",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11822",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1792",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1546",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4249",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9868",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5782",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13317",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6481",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12809",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11848",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR3128",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13044",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9326",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10165",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1339",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6213",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5067",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5840",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6637",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1993",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9164",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4870",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR7424",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10886",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6712",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13517",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11947",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11364",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9426",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2740",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12953",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6379",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR104",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5621",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6039",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6474",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR11684",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3572",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8354",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11200",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5657",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR11432",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR754",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10651",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4227",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2552",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9778",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3341",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4530",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4217",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2563",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2170",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3689",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8457",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR10566",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1275",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3719",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3430",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10034",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13313",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12759",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12912",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Construction road.",
        "frame": "FR1542",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5349",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13358",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7508",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR10554",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12861",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6633",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR823",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5073",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11135",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11372",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7304",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13595",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR501",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2769",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2730",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13386",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2312",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR664",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9245",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8105",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4343",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5375",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11260",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4824",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1014",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12853",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR7943",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR8310",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12142",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4945",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9350",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR8794",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4630",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR359",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12272",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR13348",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR5899",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR1568",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR6211",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3423",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR4583",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5623",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR4703",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3692",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9490",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR11765",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR2590",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9304",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR13708",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR1419",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR12154",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR5537",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Undeveloped road.",
        "frame": "FR234",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR3576",
        "question": "What kind of road scene is it in the images?"
    },
    {
        "answer": "Normal city road.",
        "frame": "FR9996",
        "question": "What kind of road scene is it in the images?"
    }
]
