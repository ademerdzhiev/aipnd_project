# PROGRAMMER: Angel
# DATE CREATED: 25.05.2020
# REVISED DATE: 30.05.2020
# PURPOSE: Create a functions for preparing the data for processing, processing and displaying images
#

import torch
import torchvision
from torchvision import transforms
import numpy as np
from input_args import args_input
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image


def dataloader_datasets():
    """
        1. Gets the folders where the training, validation and testing data is located.
        2. Torchvision transforms are used to augment the training data with random scaling,
            rotations, mirroring, and/or cropping.
        3. The training, validation, and testing data is appropriately cropped and normalized.
        4. The data for each set (train, validation, test) is loaded with torchvision's ImageFolder.
        5. The data for each set is loaded with torchvision's DataLoader.
    """
    in_arg = args_input()
    data_dir = in_arg.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

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
    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    datasets = [train_data, valid_data, test_data]
    dataloader = [trainloader, validloader, testloader]

    return dataloader, datasets


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    :param image: the image to be processed
    :return: Numpy array
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im_pil = Image.open(image)
    im_pil = im_pil.resize((256, 256))
    crop_value = 0.5 * (256 - 224)
    im_pil = im_pil.crop((crop_value, crop_value, 256 - crop_value, 256 - crop_value))
    im_pil = np.array(im_pil) / 255
    im_pil = (im_pil - mean) / std

    return im_pil.transpose((2, 0, 1))


def display_image(probs, labels, img_path):
    """
    A matplotlib figure is created displaying an image and its associated top k most
    probable classes with actual flower names.
    :param probs: a list with the top k probabilities
    :param labels: a list with the top k labels
    :param img_path: path to the image to be displayed
    """
    # cropping image to have the desired size of 256x256
    img = plt.imread(img_path)
    img = resize(img, (256, 256), mode='constant')
    new_height1 = int(img.shape[0] / 2 - 224 / 2)
    new_height2 = int(img.shape[0] / 2 + 224 / 2)
    new_width1 = int(img.shape[1] / 2 - 224 / 2)
    new_width2 = int(img.shape[1] / 2 + 224 / 2)
    img_cropped = img[new_height1:new_height2, new_width1:new_width2]

    # plotting the results

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
    ax1.set_title(labels[0])
    ax1.set_title(labels[0])
    ax1.axis('off')
    ax1.imshow(img_cropped)

    y_pos = np.arange(5)
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Probability')
    plt.show()