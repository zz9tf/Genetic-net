import pathlib
import os
import yaml
import argparse
from easydict import EasyDict as edict

def manually_set_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default=None,
                        help='the target task to execute')
    # detaset
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='name of dataset: movielens, yelp, census_income, churn, fraud_detection')
    # basic configs
    parser.add_argument('--model', type=str, default=None,
                        help='The model to be used (Now you have: linear_regression, etc...)')
    # experiment
    parser.add_argument('--task_num', type=int, default=None,
                        help='tell experiment method which task part to execute')
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='the path for experiments to save result')

    return parser.parse_args()

def loading_args():
    args = edict(vars(manually_set_configs()))
    configs_dir = pathlib.Path(__file__).parent.absolute()
    print()
    for file in os.listdir(configs_dir):
        if file.endswith('.yml'):
            args.update(yaml.safe_load(open(os.path.join(configs_dir, file), 'r')))
    
    args.datapath = os.path.join(os.path.abspath(configs_dir.parent.parent), args.datapath)
    args.checkpoint_dir = os.path.join(os.path.abspath(configs_dir.parent.parent), args.checkpoint_dir)
    args.experiment_dir = os.path.join(os.path.abspath(configs_dir.parent.parent), args.experiment_dir)
    
    return args
