# PROGRAMMER: Angel
# DATE CREATED: 25.05.2020
# REVISED DATE: 26.05.2020
# PURPOSE: Predicts the top k classes for a given image and displays it.

from model_fc import predict
from checkpoint import load_checkpoint
from input_args import args_input
from utility_functions import display_image
import json


args = args_input()
img_path = args.image_path
checkpoint_path = args.checkpoint


def main():
    model, optimizer = load_checkpoint(checkpoint_path)
    probs, index = predict(img_path, model)

    print(probs)

    if args.category_names != '':
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

        labels = []
        for n in range(len(index)):
            labels.append(cat_to_name[str(index[n])])

        print(labels)
        # uncomment the line below to display the image and to plot the top k probabilities
        # display_image(probs, labels, img_path)

    else:
        print(index)
       # uncomment the line below to display the image and to plot the top k probabilities
       # display_image(probs, index, img_path)


if __name__== "__main__":
    main()