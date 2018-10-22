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



def load_data(input_dir ):
    train_dir = input_dir + '/train'
    validation_dir = input_dir + '/valid'
    test_dir = input_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    #No random operations
    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    #load image datasets with image folder
    train_img_ds = datasets.ImageFolder(input_dir +'/train', transform = train_transforms)
    val_img_ds = datasets.ImageFolder(input_dir + '/valid', transform = validation_transforms)
    test_img_ds = datasets.ImageFolder(input_dir + '/test', transform = test_transforms)

     # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader= torch.utils.data.DataLoader(train_img_ds, batch_size = 64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_img_ds, batch_size = 32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_img_ds, batch_size = 32, shuffle=True)

    # first
    print("Train Images:",  len(train_img_ds.imgs), "Train Labels:", len(train_img_ds.classes))
    print("Test Images:",  len(val_img_ds.imgs), "Test Labels:", len(val_img_ds.classes))
    print("Validation Images:",  len(test_img_ds.imgs), "Validation Labels:", len(test_img_ds.classes))
    return train_dataloader, val_dataloader, test_dataloader, train_img_ds.class_to_idx

def validation(model, dataloader,criterion):
    loss = 0
    accuracy = 0
    for (inputs, labels) in dataloader:
        inputs, labels= inputs.to('cuda'), labels.to('cuda')

        #forward
        output = model.forward(inputs)
        loss += criterion(output,labels).item()

        ps = torch.exp(output)
        # grab the maximum probability for each tensor and compare to the associated label.
        # The result here will be a tensor of 0's and 1's.
        equality = (labels.data == ps.max(dim=1)[1])
        #Convert equality tensor to a float, take mean to determine accuracy
        accuracy += equality.type(torch.FloatTensor).mean()


    return loss, accuracy

def build_model(structure = 'vgg19', hidden_layers = [4096,2048, 512], output_size = 102, dropout = 0.25, lr = 0.001, engine='gpu'):

    arch = {"vgg19":25088,"densenet121":1024,"alexnet":9216}

    if structure == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a valid model.Did you mean vgg19,densenet121,or alexnet?".format(structure))

    for param in model.parameters():
        param.requires_grad = False

        classifier = nn.Sequential(
            nn.Linear(arch[structure], hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layers[2], output_size),
            nn.LogSoftmax(dim=1))

        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )

        if torch.cuda.is_available() and engine == 'cuda':
          model.cuda()

        return model, criterion, optimizer

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs = 8, print_every = 10, engine ='cuda'):

    steps = 0
    model.to(engine)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for (inputs,labels) in train_dataloader:
            steps +=1
            inputs, labels = inputs.to(engine), labels.to(engine)
            #ensure gradients are not carried forward
            optimizer.zero_grad()
            #forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            #update weights
            loss.backward()
            optimizer.step()
            #update loss
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, val_accuracy = validation(model, val_dataloader, criterion)
                print(
                    "Epoch {}/{}".format(e+1, epochs),
                    "Training Loss {:.3f}".format(running_loss/print_every),
                    "Val Acc {:.3f}".format(val_accuracy/len(val_dataloader)),
                    "Val Loss {:.3f}".format(val_loss/len(val_dataloader))
                                        )

                running_loss = 0
                model.train()

def test_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            output = model(inputs)
            _, prediction = torch.max(output.data, dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
            accuracy = 100 * correct/total
        print("Accuracy {:.2f}".format(accuracy))

def save_model(model, epochs, optimizer, train_class_idx):
    '''
    Arguments: All arguements are inputs to building a model. These must be stored so we can rebhild the model when reloading at a later time.
    Returns: This function does not return anything. It simply saves the model at the desired directory.
    '''

    model.class_to_idx = train_class_idx
    checkpoint = { 'state_dict': model.state_dict(),
                  'image_datasets' : model.class_to_idx,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'model': model }
    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(save_dir):
    '''
    Arguemnts: save_dir is the directory where the checkpoints are located
    Returns: function rebuilds the model based and uses the state_dict to return to trained weights
    '''
    checkpoint = torch.load(save_dir)
    model,_,_ = build_model()

    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['image_datasets']
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path) # Here we open the image

    make_img_good = transforms.Compose([ # Here as we did with the traini ng data we will define a set of
        # transfomations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = make_img_good(img)

    return tensor_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, topk=5, engine='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and engine=='gpu':
        model.to('cuda')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if engine == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)
