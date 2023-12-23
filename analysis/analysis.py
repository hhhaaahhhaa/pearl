import pickle
import logging
import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional
import pickle

from actor import Actor
from env import Env
from llm import LLM
from datamodule import DataModule
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import matplotlib.pyplot as plt

# from eval_agents import *

# huggingface_hub.login(token="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")

prompt_to_id = {
    "Analyze the given information, break down the problem into manageable steps, apply suitable mathematical operations, and provide a clear, accurate, and concise solution, ensuring precise rounding if necessary. Consider all variables and carefully consider the problem’s context for an efficient solution.": 0,
    "Answer Directly.": 1,
    "Break this down.": 2,
    "Embrace challenges as opportunities for growth. Each obstacle you overcome brings you closer to success.": 3,
    'Let’s be realistic and think step by step.': 4,
    'Let’s solve this problem by splitting it into steps.': 5,
    'Let’s think about this logically.': 6,
    'Let’s think like a detective step by step.': 7,
    'Let’s think step by step.': 8,
    'Let’s work this out in a step by step way to be sure we have the right answer.': 9,
    'Let’s work through this problem step-by-step:': 10,
    'Question decomposition.': 11,
    'Remember that progress is made one step at a time. Stay determined and keep moving forward.': 12,
    'Stay focused and dedicated to your goals. Your consistent efforts will lead to outstanding achievements.': 13,
    'Take a deep breath and work on this problem step-by-step.': 14,
    'Take a deep breath and work on this problem.': 15,
    'Take pride in your work and give it your best. Your commitment to excellence sets you apart.': 16,
    'This is very important to my career.': 17,
    'Write your answer and give me a confidence score between 0-1 for your answer.' : 18,
    'You have to solve this problem, I am in trouble.': 19,
    "You'd better be sure.": 20
}

def read_data(file_path):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data


def get_token_length(tokenizer, text):
    return len(tokenizer(text)['input_ids'])


# Calculate for each prompt, the correctness and its response length.
def evaluate_each_prompt(test_dataset, pickle_result_dir, output_result_dir):

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for prompt_pool in tqdm(prompt_to_id):

        stats = {}

        correct = 0
        total = 0

        prompt_pool_id = prompt_to_id[prompt_pool]
        print(f"Processing {prompt_pool_id}: {prompt_pool}")
        
        pickle_result = read_data(file_path=f"{pickle_result_dir}/prompt-{prompt_pool_id}.pkl")

        for order, (result, ground_truth) in enumerate(zip(pickle_result, test_dataset)):
            
            # ground_truth.
            answer = ground_truth["answer"]
            options = ground_truth["options"]
            answer_index = options.index(answer)

            # llama output.
            confidence_score = result["scores"]
            confidence_score_index = np.argmax(confidence_score)
            # print(f"Confidence Score Index: {confidence_score_index}")

            # Check whether correct and its response length.
            if confidence_score_index == answer_index:
                prompt_response = (ground_truth["prompt"][prompt_pool])
                prompt_response_length = get_token_length(tokenizer, prompt_response)
                stats[order] = prompt_response_length
                correct += 1
            
            # The answer is not in the options.
            else:
                stats[order] = 666666
            
            total += 1
        
        print(f"PromptID: {prompt_pool_id}, Acc = {round((correct / total), 3)}")

        output_stats_path = f"{output_result_dir}/stats-{prompt_pool_id}.json"

        with open(output_stats_path, 'w') as f:
            json.dump(stats, f, indent=4)


def analysis_best_prompt(split_name, stats_result_dir, graph_result_dir):
    
    metadata = {}

    for stat_json in os.listdir(stats_result_dir):
        with open(f"{stats_result_dir}/{stat_json}") as f:
            data = json.load(f)
            prompt_id = stat_json.split(".")[0].split("-")[1]
            
            for index in data:
                if index not in metadata:
                    metadata[index] = []
                metadata[index].append(data[index])
    
    stats = {}
    no_correct_answer = 0
    for order, test_sample in enumerate(metadata):

        # split-1, openbookqa_test
        if split_name == "openbookqa_test":
            if order >= 500: break

        # split-2, strategyqa_test
        elif split_name == "strategyqa_test":
            if order < 500: continue
            if order >= 1000: break

        # split-3, truthfulqa_test
        elif split_name == "truthfulqa_test":
            if order < 1000: continue
    
        every_prompt_result = metadata[test_sample]
        best_prompt_index = np.argmin(every_prompt_result)
        # print(best_prompt_index)
        if every_prompt_result[best_prompt_index] == (666666):
            # print("No Answer")
            no_correct_answer += 1
        else:
            if best_prompt_index not in stats:
                stats[best_prompt_index] = 0
            stats[best_prompt_index] += 1

    # print(stats)
    print(f"No Answer: {no_correct_answer}")
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(16, 8))
    plt.bar(stats.keys(), stats.values())
    plt.title(f'Bar Chart: {split_name}')
    plt.xlabel('PromptID')
    plt.ylabel('Correct Num')
    plt.savefig(f"{graph_result_dir}/bar_chart-{split_name}-overall.png")

    plt.figure(figsize=(16, 16))
    plt.pie(stats.values(), labels=stats.keys(), autopct='%1.1f%%')
    plt.title(f'Pie Chart: {split_name}')
    plt.savefig(f"{graph_result_dir}/pie_chart-{split_name}-overall.png")


# Calculate for each prompt, the correctness and its response length.
def main_analysis_best_prompt():

    pickle_result_dir = "/work/u2619111/frank/rl_final/pearl/result_v2_revision/logs"
    output_result_dir = "/work/u2619111/frank/rl_final/pearl/result_v3/stats"
    stats_result_dir = output_result_dir
    graph_result_dir = "/work/u2619111/frank/rl_final/pearl/result_v3/graph_result"

    # "openbookqa_test", "strategyqa_test", "truthfulqa_test"

    datamodule = DataModule()
    test_dataset = datamodule.test_dataset
    evaluate_each_prompt(test_dataset, pickle_result_dir, output_result_dir)

    for split_name in ["openbookqa_test", "strategyqa_test", "truthfulqa_test"]:
        analysis_best_prompt(split_name, stats_result_dir, graph_result_dir)


def analysis_flip(test_dataset, split_name, no_prompt_pickle_data_path, pickle_result_dir, output_result_dir, output_graph_dir):

    # "openbookqa_test", "strategyqa_test", "truthfulqa_test"
    # split_name = "openbookqa_test"
    # split_name = "strategyqa_test"
    # split_name = "truthfulqa_test"

    no_prompt_result = read_data(file_path=no_prompt_pickle_data_path)

    stats = {}

    for index, prompt_pool in tqdm(enumerate(prompt_to_id)):

        correct = 0
        total = 0

        prompt_pool_id = prompt_to_id[prompt_pool]
        print(f"Processing {prompt_pool_id}: {prompt_pool}")
        
        pickle_result = read_data(file_path=f"{pickle_result_dir}/prompt-{prompt_pool_id}.pkl")

        for order, (result, no_prompt, ground_truth) in enumerate(zip(pickle_result, no_prompt_result, test_dataset)):
            
            # split-1
            if split_name == "openbookqa_test":
                if order >= 500: break

            # split-2
            elif split_name == "strategyqa_test":
                if order < 500: continue
                if order >= 1000: break

            # split-3
            elif split_name == "truthfulqa_test":
                if order < 1000: continue

            # ground_truth.
            answer = ground_truth["answer"]
            options = ground_truth["options"]
            answer_index = options.index(answer)

            # no prompt result.
            no_prompt_score = no_prompt["scores"]
            no_prompt_score_index = np.argmax(no_prompt_score)
            # print(f"Answer Index: {answer_index}")

            # llama output.
            confidence_score = result["scores"]
            confidence_score_index = np.argmax(confidence_score)
            # print(f"Confidence Score Index: {confidence_score_index}")

            # Check whether correct and its response length.
            # correct to wrong
            if no_prompt_score_index == answer_index and no_prompt_score_index != confidence_score_index:
                if index not in stats:
                    stats[index] = {
                        "correct_to_wrong": 0,
                        "wrong_to_correct": 0
                    }
                stats[index]["correct_to_wrong"] += 1

            # wrong to correct
            if no_prompt_score_index != answer_index and answer_index == confidence_score_index:
                if index not in stats:
                    stats[index] = {
                        "correct_to_wrong": 0,
                        "wrong_to_correct": 0
                    }
                stats[index]["wrong_to_correct"] += 1
                    
        # print(f"PromptID: {prompt_pool_id}, Acc = {round((correct / total), 3)}")
    output_stats_path = f"{output_result_dir}/stats-flip-{split_name}-x-o.json"
    with open(output_stats_path, 'w') as f:
        json.dump(stats, f, indent=4)

    data = stats

    # Extracting data for plotting
    plt.rcParams.update({'font.size': 20})
    indexes = list(data.keys())
    correct_to_wrong = [data[index]["correct_to_wrong"] for index in indexes]
    wrong_to_correct = [data[index]["wrong_to_correct"] for index in indexes]

    # Creating a bar plot
    plt.figure(figsize=(15, 8))
    bar_width = 0.35
    index = range(len(indexes))

    # Plotting two bars for each index
    plt.bar(index, correct_to_wrong, bar_width, label='Correct to Wrong', color='blue')
    plt.bar([i + bar_width for i in index], wrong_to_correct, bar_width, label='Wrong to Correct', color='red')

    # Adding labels and title
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.title('Comparison of Correct to Wrong and Wrong to Correct Counts per Index')
    plt.xticks([i + bar_width / 2 for i in index], indexes)
    plt.legend()

    # Displaying the plot
    plt.tight_layout()
    plt.savefig(f"{output_graph_dir}/bar_chart-flip-{split_name}.png")


def main_analysis_flip():

    no_prompt_pickle_data_path = f"/work/u2619111/frank/rl_final/pearl/result_v2_revision/no-prompt-log.pkl"
    pickle_result_dir = "/work/u2619111/frank/rl_final/pearl/result_v2_revision/logs"
    output_result_dir = "/work/u2619111/frank/rl_final/pearl/result_v3/stats"
    output_graph_dir = "/work/u2619111/frank/rl_final/pearl/result_v3/graph_result"

    datamodule = DataModule()
    test_dataset = datamodule.test_dataset

    for split_name in ["openbookqa_test", "strategyqa_test", "truthfulqa_test"]:
        analysis_flip(test_dataset, split_name, no_prompt_pickle_data_path, pickle_result_dir, output_result_dir, output_graph_dir)


if __name__ == '__main__':
    
    # Analysis the best prompt for each sample.
    main_analysis_best_prompt()


    # Analysis the flip.
    main_analysis_flip()
