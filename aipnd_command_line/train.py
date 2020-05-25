import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import model_fc
import train_args

in_arg = train_args.args_input()
device = in_arg.gpu
data_dir = in_arg.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

dataloader = [trainloader, validloader]

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

model_fc.train(model, trainloader, validloader, criterion, optimizer)


