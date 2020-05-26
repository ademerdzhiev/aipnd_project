# PROGRAMMER: Angel
# DATE CREATED: 24.05.2020
# REVISED DATE: 26.05.2020
# PURPOSE: Create a function that retrieves the following 3 command line inputs
#          from the user using the Argparse Python module. If the user fails to
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:

import argparse
import torch


def args_input():
    """
       Retrieves and parses command line arguments provided by the user when
       they run the program from a terminal window. This function uses Python's
       argparse module to create and define these command line arguments. If
       the user fails to provide some or all of the arguments, then the default
       values are used for the missing arguments.
       Command Line Arguments:
         1. Set directory to save checkpoints as --save_dir save_directory
         2. CNN Model Architecture as --arch "vgg13"
         3. Set hyperparameters as  --learning_rate 0.01 --hidden_units 512 --epochs 20
         4. Use GPU for training as --gpu
       This function returns these arguments as an ArgumentParser object.
       Parameters:
        None - simply using argparse module to create & store command line arguments
       Returns:
        parse_args() -data structure that stores the command line arguments object
       """

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', default='./flowers/test/1/image_06743.jpg', nargs='?', action="store",
                         help='path to the image we want to process with predict.py')

    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store",
                         help='path to the saved checkpoint of the model')

    parser.add_argument('--top_k', type=int, default=5,
                         help='the top K most likely classes')

    parser.add_argument('--category_names', default='', type=str, action='store')

    parser.add_argument('--dir', type=str, default='flowers/',
                        help='path to the data folder')

    parser.add_argument('--save_dir', type=str, default='./',
                        help='path to the folder in which the checkpoint is saved')

    parser.add_argument('--arch', type=str, default='vgg19',
                        help='the CNN model architecture')

    parser.add_argument('--hidden_units', nargs='+',
                        help='list with the hidden layers')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='the learnirning rate')

    parser.add_argument('--drop_out', type=float, default='0.3')

    parser.add_argument('--epochs', type=int, default=24,
                        help='the number of epochs')

    parser.add_argument('--gpu', type=str, default='cpu', help='using GPU for training')

    args = parser.parse_args()

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.gpu = device
        print('using {} for training'.format(device))

    return args
