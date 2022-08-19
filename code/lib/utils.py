'''
This python file is used to save special methods in the research

'''
from pathlib import Path
import torch
import numpy as np
import os
from scipy.stats import pearsonr
from math import ceil

def check_path(args):
    """
    This method checks all folder paths used in the whole program and creates a new one
    if it is used for saving results and doesn't exist.

    Args:
        args (_type_): _description_
    """
    '''
    
    '''
    assert os.path.exists(args.datapath)
    
    if os.path.exists(Path(args.checkpoint_dir).parent) is False:
        os.mkdir(Path(args.checkpoint_dir).parent)
    if os.path.exists(args.checkpoint_dir) is False:
        os.mkdir(args.checkpoint_dir)
    
    if os.path.exists(Path(args.experiment_dir).parent) is False:
        os.mkdir(Path(args.experiment_dir).parent)
    if os.path.exists(args.experiment_dir) is False:
        os.mkdir(args.experiment_dir)