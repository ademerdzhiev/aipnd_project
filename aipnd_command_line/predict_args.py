# PROGRAMMER: Angel
# DATE CREATED: 26.05.2020
# REVISED DATE:

import argparse


def predict_input_args():
    parser_ = argparse.ArgumentParser(description='./predict.py')

    parser_.add_argument('image_path', action="store", type=str,
                        help='path to the image we want to process with predict.py')

    parser_.add_argument('checkpoint', action="store", type=str,
                        help='path to the saved checkpoint of the model')

    parser_.add_argument('--top_k', type=int, default=5,
                        help='the top K most likely classes')
    return parser_.parse_args()
