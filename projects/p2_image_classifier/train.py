import torch
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms, models
import time
import json
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import utils

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments


ap.add_argument('--engine', default="cuda")
ap.add_argument('--hidden_layers', type=int, nargs='+', default=[4096,2048, 512], help='Number of units per hidden layer')
ap.add_argument('--output_size', type=int, default=102, help='Number of possible classifications')
ap.add_argument('--lr', type=float, action="store", default=0.001)
ap.add_argument('--dropout', type=float, action = "store", default = 0.25)
ap.add_argument('--epochs', action="store", type=int, default = 1)
ap.add_argument('--arch', action="store", default="vgg19", type = str)
ap.add_argument('--print_every', action='store', default=10 )
ap.add_argument('--input_dir', nargs='*', action="store", default="flowers")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")

args = ap.parse_args()

train_dataloader, val_dataloader, test_dataloader, class_to_idx_train = utils.load_data(args.input_dir)


model, criterion, optimizer = utils.build_model(args.arch,args.hidden_layers,args.output_size,args.dropout,args.lr, args.engine)


utils.train_model(model, train_dataloader, val_dataloader, criterion, optimizer, args.epochs, args.print_every, args.engine)

utils.test_model(model, test_dataloader)

utils.save_model(model, args.epochs, optimizer, class_to_idx_train)


print("All Done. The Model is trained")
