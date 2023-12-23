import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def get_data(path):
    with open(path) as f:
        data = json.load(f)
    return data


def main(args):
    output_dir = args.output_stats
    split_name = args.split_name

    data = get_data(f"{output_dir}/fix/{split_name}.json")
    random_data = get_data(f"{output_dir}/random/{split_name}.json")
    topline_data = get_data(f"{output_dir}/topline/{split_name}.json")
    reward_v2_data = get_data(f"{output_dir}/reward_v2/{split_name}.json")
    reward_v3_data = get_data(f"{output_dir}/reward_v3/{split_name}.json")
    reward_v4_data = get_data(f"{output_dir}/reward_v4/{split_name}.json")
    reward_v5_data = get_data(f"{output_dir}/reward_v5/{split_name}.json")

    os.makedirs(args.output_plot, exist_ok=True)
    output_plot = f"{args.output_plot}/{split_name}.jpg"

    labels = list(data.keys())
    acc_values = [data[key]['acc'] for key in data]
    length_values = [data[key]['length'] for key in data]
    # acc_values.append(random_data['acc'])
    # acc_values.append(topline_data['acc'])
    # length_values.append(random_data['length'])
    # length_values.append(topline_data['length'])

    # Create a scatter plot with 'x' markers for each data point
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(length_values, acc_values, marker='x', color='blue')
    plt.scatter([random_data['length']], [random_data['acc']], marker='x', color='red')
    # plt.scatter([topline_data['length']], [topline_data['acc']], marker='x', color='red')
    plt.scatter([reward_v2_data['length']], [reward_v2_data['acc']], marker='x', color='red')
    plt.scatter([reward_v3_data['length']], [reward_v3_data['acc']], marker='x', color='red')
    plt.scatter([reward_v4_data['length']], [reward_v4_data['acc']], marker='x', color='red')
    plt.scatter([reward_v5_data['length']], [reward_v5_data['acc']], marker='x', color='red')

    # Adding title and labels
    # plt.title('Accuracy vs Token Length Distribution')
    plt.xlabel('Token Length')
    plt.ylabel('Accuracy')

    for prompt, result in data.items():
        
        plt.text(result['length'], result['acc'], prompt[:15] + "...", fontsize=6, ha='left', va='bottom')
    
    plt.text(random_data['length'], random_data['acc'], "Random", fontsize=8, ha='left', va='bottom', color="red")
    # plt.text(topline_data['length'], topline_data['acc'], "Topline", fontsize=6, ha='left', va='bottom')
    # 1 / (correct_score * length): Entropy
    plt.text(reward_v2_data['length'], reward_v2_data['acc'], "Entropy", fontsize=8, ha='left', va='bottom', color="red")
    # correct_score: Score
    plt.text(reward_v3_data['length'], reward_v3_data['acc'], "Score", fontsize=8, ha='left', va='bottom', color="red")
    # correct_score / length: Simple
    plt.text(reward_v4_data['length'], reward_v4_data['acc'], "Simple", fontsize=8, ha='left', va='bottom', color="red")
    # 1 / length: Length
    plt.text(reward_v5_data['length'], reward_v5_data['acc'], "Length", fontsize=8, ha='left', va='bottom', color="red")

    plt.grid()
    plt.savefig(output_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_name', type=str, default='hotpotqa_validation')
    # parser.add_argument('--split_name', type=str, default='openbookqa_test')
    # parser.add_argument('--split_name', type=str, default='strategyqa_test')
    # parser.add_argument('--split_name', type=str, default='truthfulqa_test')
    parser.add_argument('--output_stats', type=str, default='./_data/analysis')
    parser.add_argument('--output_plot', type=str, default='./_data/analysis/tradeoff')
    args = parser.parse_args()
    main(args)
