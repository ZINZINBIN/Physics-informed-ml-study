import numpy as np
import os
import random
import argparse
import requests
import torch
from torch.utils.data import Dataset

print("torch version {}".format(torch.__version__))

parser = argparse.ArgumentParser(description="download data from external")
parser.add_argument("--dir", type = str, default = "./src/data/")
args = vars(parser.parse_args())

# get training data
dir = args["dir"]

if not os.path.isfile(dir + 'data-airfoils.npz'): 
    print("Downloading training data (300MB), this can take a few minutes the first time...") 
    with open(dir + "data-airfoils.npz", 'wb') as datafile:
        resp = requests.get('https://dataserv.ub.tum.de/s/m1615239/download?path=%2F&files=dfp-data-400.npz', verify=False) 
        datafile.write(resp.content)