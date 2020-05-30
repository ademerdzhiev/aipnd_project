# PROGRAMMER: Angel
# DATE CREATED: 25.05.2020
# REVISED DATE: 26.05.2020
# PURPOSE: Creates functions for saving and loading a checkpoint

import torch
from torch import optim
from torchvision import models
import utility_functions
from input_args import args_input

in_arg = args_input()


def save_checkpoint(model, optimizer, epochs=in_arg.epochs, save_dir=in_arg.save_dir):
    """
    Saving a checkpoint - the number of epochs, the learning rate, the classifier, the state of the model,
    the optimizer, the class to index dictionary.
    :param model: passing the trained model
    :param optimizer: the backpropagation optimizer
    :param epochs: number of epochs the models was trained
    :param save_dir: the directory where the checkpoint is to be saved
    :return:
    """
    dataloader, datasets = utility_functions.dataloader_datasets()
    model.class_to_idx = datasets[0].class_to_idx

    checkpoint = {'epochs': epochs,
                  'arch': 'vgg19',
                  'learning_rate': in_arg.learning_rate,
                  'classifier': model.classifier,
                  'model_state': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, save_dir + 'checkpoint.pth')
    print('checkpoint saved')
    print(checkpoint['model_state'])
    print(checkpoint['classifier'])


def load_checkpoint(filepath):
    """
    Loads the saved model.
    :param filepath: the path to the saved checkpoint
    :return: tuple containing the trained model and the optimizer
    """
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