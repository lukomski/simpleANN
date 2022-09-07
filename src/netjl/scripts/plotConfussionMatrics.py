import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from collections import deque


def saveMatrix(cm, labels, output_file):
    number_of_classes = len(labels)
    df_cm = pd.DataFrame(cm, range(number_of_classes),
                         range(number_of_classes))

    ax = plt.subplot()
    sns.set(font_scale=1)
    # annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(df_cm, annot=True, fmt='g', ax=ax, cmap="YlGnBu")

    # labels, title and ticks
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Expected class')
    ax.set_title("Confussion Matrix")
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # plt.show()
    print("save", output_file)
    plt.savefig(output_file)


parser = argparse.ArgumentParser()
parser.add_argument('--metrics_file', type=str,
                    help='Metrics JSON file')
parser.add_argument('--config_file', type=str,
                    help='Config JSON file')
parser.add_argument('--out_files_directory_path', type=str,
                    help='Result files directory path')

args = parser.parse_args()

json_file = args.metrics_file
out_files_directory_path: str = args.out_files_directory_path

if not os.path.exists(out_files_directory_path):
    os.makedirs(out_files_directory_path)
    print("The new directory is created!")


f = open(args.config_file, "r")
config = json.loads(f.read())
f.close()

labels = config["classes"]
listA = deque(labels)
listA.appendleft("0")
labels = list(listA)

# Read metrics
f = open(args.metrics_file, "r")
data = json.loads(f.read())
f.close()

for idx, cm_test in enumerate(data['Confussion matrix']['test']):
    output_file = f'{out_files_directory_path}cm_test_{idx}.png'
    saveMatrix(cm_test, labels, output_file)

for idx, cm_train in enumerate(data['Confussion matrix']['train']):
    output_file = f'{out_files_directory_path}cm_train_{idx}.png'
    saveMatrix(cm_train, labels, output_file)
