# PROGRAMMER: Angel
# DATE CREATED: 24.05.2020
# REVISED DATE: 26.05.2020
# PURPOSE: Create a function that retrieves command line inputs from the user using
#          the Argparse Python module.

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

        To be passed to train.py:
         1. Set directory to save checkpoints as --save_dir save_directory
         2. CNN Model Architecture as --arch 'vgg19'
         3. Set hyperparameters as  --learning_rate 0.0001 --hidden_units 512 --epochs 24
         4. Use GPU for training as --gpu
         5. Dropout as --drop_out 0.3

        To be passed to predict.py:
         6. top k classes as --top_k 5
         7. the category names as --category names
         8. loading saved model as checkpoint './checkpoint.pth'
         9. image path to image in order to predict to which class it belongs
            image_path './flowers/test/1/image_06743.jpg'

       This function returns these arguments as an ArgumentParser object.
       Parameters:
        None - simply using argparse module to create & store command line arguments
       Returns:
        parse_args() -data structure that stores the command line arguments object
       """

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', default='./flowers/test/11/image_03115.jpg', nargs='?', action="store",
                         help='path to the image we want to process with predict.py')

    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store",
                         help='path to the saved checkpoint of the model')

    parser.add_argument('--top_k', type=int, default=5,
                         help='the top K most likely classes')

    parser.add_argument('--category_names', default='', type=str, action='store',
                         help='the names of the categories/flowers')

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

    parser.add_argument('--gpu', action="store_true", default=False, help='using GPU for training')

    args = parser.parse_args()

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.gpu = device
        print('using {}'.format(device))
    else:
        args.gpu = "cpu"
        device = args.gpu
        print('using {}'.format(device))


    return args
