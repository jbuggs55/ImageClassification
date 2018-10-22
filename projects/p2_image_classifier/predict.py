import torch
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms, models
import json
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import utils

ap = argparse.ArgumentParser(description='Predict.py')
# Command Line arguments

ap.add_argument('--input_dir', nargs='*', action="store", default="flowers")
ap.add_argument('--input_img', default='/home/workspace/aipnd-project/flowers/test/1/image_06743.jpg')
ap.add_argument('--save_dir', default='checkpoint.pth')
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--engine', default="gpu", action="store", dest="gpu")

args = ap.parse_args()

train_dataloader, val_dataloader, test_dataloader, _ = utils.load_data(args.input_dir)

model = utils.load_checkpoint(args.save_dir)

with open(args.category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

probs= utils.predict(args.input_img, model,args.top_k)
print (probs)

labels = [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
probability = np.array(probs[0][0])

i=0
while i < args.top_k:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1
