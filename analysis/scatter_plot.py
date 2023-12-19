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

    data = get_data(path=args.data)

    statistics = {}

    for sample in tqdm(data):
        answer = sample['answer']
        answer_index = sample['options'].index(answer)

        for prompt in sample['inst']:
            if prompt not in statistics:
                statistics[prompt] = {
                    'total': 0,
                    'correct': 0,
                    'token_length': 0,
                    'accuracy': 0,
                }
            
            score = sample['inst'][prompt]['score']
            predict_answer_index = score.index(max(score))

            if predict_answer_index == answer_index:
                statistics[prompt]['correct'] += 1
                statistics[prompt]['token_length'] += (len(sample['inst'][prompt]['step'].split()))
            
            statistics[prompt]['total'] += 1
    
    for stats in statistics:
        statistics[stats]['token_length'] /= statistics[prompt]['total']
        statistics[stats]['accuracy'] = statistics[stats]['correct'] / statistics[stats]['total']


    with open(args.output_stats, 'w') as f:
        json.dump(statistics, f, indent=4)
    
    plot(statistics, args.output_plot)
    

def plot(statistics, output_plot):

    data = statistics

    labels = list(data.keys())
    acc_values = [data[key]['accuracy'] for key in data]
    length_values = [data[key]['token_length'] for key in data]

    # Create a scatter plot with 'x' markers for each data point
    plt.figure(figsize=(10, 6))
    plt.scatter(length_values, acc_values, marker='x', color='blue')

    # Adding title and labels
    plt.title('Accuracy vs Token Length Distribution')
    plt.xlabel('Token Length')
    plt.ylabel('Accuracy')

    for prompt, result in data.items():
        
        plt.text(result['token_length'], result['accuracy'], prompt[:10], fontsize=6, ha='right', va='bottom')

    plt.grid()
    plt.savefig(output_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/Users/kuanyiiii/Downloads/hotpotqa_validation_processed_data.json')
    parser.add_argument('--output_stats', type=str, default='./statistics.json')
    parser.add_argument('--output_plot', type=str, default='./accuracy_vs_token_length.png')
    args = parser.parse_args()
    main(args)