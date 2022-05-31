#!/usr/bin/env python
import os
import sys
from time import time
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import numpy as np
import pandas as pd

from davis2017.evaluation import DAVISEvaluation

import torch
import models
from configs.local_config import config

default_davis_path = '/path/to/the/folder/DAVIS'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--davis_path', type=str, help='Path to the DAVIS folder containing the JPEGImages, Annotations, '
                                                   'ImageSets, Annotations_unsupervised folders',
                    required=False, default=default_davis_path)
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--year', type=str, help='year of dataset', default='2016')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='unsupervised',
                    choices=['semi-supervised', 'unsupervised'])
parser.add_argument("--from_folder", action="store_true", dest="from_folder", help='If using stored results or not')
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    required=True)
parser.add_argument('--model_path', type=str, help='Path to the trained model',
                    required=True)                 
args, _ = parser.parse_known_args()

csv_name_global = f'global_results-{args.set}.csv'
csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)
if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences for the {args.task} task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(
        davis_root=args.davis_path, 
        year=args.year, 
        task=args.task, 
        gt_set=args.set, 
        store_results=True, 
        res_root=args.results_path
        )
    
    if os.path.exists(args.results_path) and args.from_folder:
        metrics_res = dataset_eval.evaluate(args.results_path)
    else:
        seg_model = models.DualSaptialAttentionNet()
        print('Loading trained model %s ...'%(args.model_path))
        seg_checkpoint_dict = torch.load(args.model_path)
        seg_model.load_state_dict(seg_checkpoint_dict['seg_net'])
        seg_model.cuda()
        seg_model.eval()
        print("Pre-trained DSANet %s is loaded."%(args.model_path))
        metrics_res = dataset_eval.evaluate(seg_model, res_path=None)

    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
