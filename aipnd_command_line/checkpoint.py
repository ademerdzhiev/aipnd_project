# PROGRAMMER: Angel
# DATE CREATED: 25.05.2020
# REVISED DATE: 26.05.2020

import torch
from torch import optim
from torchvision import models
import utility_functions
from input_args import args_input

in_arg = args_input()


def save_checkpoint(model, optimizer, epochs=in_arg.epochs, save_dir=in_arg.save_dir):
    dataloader, datasets = utility_functions.dataloader_datasets()
    model.class_to_idx = datasets[0].class_to_idx

    checkpoint = {'epochs': epochs,
                  'arch': 'vgg19',
                  'learning_rate': 0.0001,
                  'classifier': model.classifier,
                  'model_state': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, save_dir + 'checkpoint.pth')
    print('checkpoint saved')
    print(checkpoint['model_state'])
    print(checkpoint['classifier'])


def load_checkpoint(filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = in_arg.learning_rate
    checkpoint = torch.load(filepath, map_location=device)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model, optimizer