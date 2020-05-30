# PROGRAMMER: Angel
# DATE CREATED: 24.05.2020
# REVISED DATE: 26.05.2020
# PURPOSE: Trains a pretrained model, saves a checkpoint and tests how it performs on a unknown data set

from torch import nn
from torch import optim
from torchvision import models
import model_fc
import utility_functions
import checkpoint
from input_args import args_input

in_arg = args_input()

device = in_arg.gpu
dataloader, dataset = utility_functions.dataloader_datasets()

if in_arg.arch == 'vgg19':
    input_size = 25088
    model = models.vgg19(pretrained=True)
elif in_arg.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_size = 1024

for param in model.parameters():
    param.requires_grad = False

model.to(device)
classifier = model_fc.classifier(input_size)

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=in_arg.learning_rate)

model_fc.train(model, dataloader[0], dataloader[1], criterion, optimizer)
checkpoint.save_checkpoint(model, optimizer)
model_fc.model_testing(model, dataloader[0], dataloader[2], criterion, optimizer)
