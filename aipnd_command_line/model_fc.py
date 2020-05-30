# PROGRAMMER: Angel
# DATE CREATED: 24.05.2020
# REVISED DATE: 30.05.2020
# PURPOSE: Create a functions for preparing the data for processing, processing and displaying images


import torch
from torch import nn
from input_args import args_input
import time
from collections import OrderedDict
from utility_functions import process_image

in_arg = args_input()

def classifier(input_size, output_size=102, hidden_layers_=in_arg.hidden_units, drop_p=in_arg.drop_out):
    """
    Feedforward network is defined for use as a classifier using the features as input. An ordered list containing
    each layer of the network is created. Then it is passed to the
    :param input_size: the input size for the classifier
    :param output_size: the number of output classes
    :param hidden_layers_: a list with the hidden layers of the classifier
    :param drop_p: the dropout parameter
    :return: returns a classifier to be applied to the pretrained network.
    """
    hidden_layers = [int(hidden_layers_[s]) for s in range(len(hidden_layers_))]

    sequential_list = []
    sequential_list.append((str(0), nn.Linear(input_size, hidden_layers[0])))
    sequential_list.append((str(1), nn.Dropout(drop_p)))
    sequential_list.append((str(2), nn.ReLU()))

    if len(hidden_layers) == 1:
        sequential_list.append((str(3), nn.Linear(hidden_layers[0], output_size)))
        sequential_list.append((str(4), nn.LogSoftmax(dim=1)))
    elif len(hidden_layers) >= 2:
        for n in range(len(hidden_layers) -1):
            sequential_list.append((str(int(n+3)), nn.Linear(hidden_layers[n], hidden_layers[n + 1])))
            sequential_list.append((str(int(n+4)), nn.Dropout(drop_p)))
            sequential_list.append((str(int(n+5)), nn.ReLU()))

        sequential_list.append((str(int(len(sequential_list) +1)), nn.Linear(hidden_layers[len(hidden_layers) - 1], output_size)))
        sequential_list.append((str(int(len(sequential_list) +2)), nn.LogSoftmax(dim=1)))

    return nn.Sequential(OrderedDict(sequential_list))


def train(model, trainloader, testloader, criterion, optimizer, epochs=in_arg.epochs, device=in_arg.gpu):
    """
    The training function
    :param model: passing a pretrained model
    :param trainloader: training data set
    :param testloader: testing data set
    :param criterion: the loss criterion
    :param optimizer: the backpropagation optimizer
    :param epochs: number of study epochs
    :param device: the device to be used in the training process (gpu or cpu)
    """
    running_loss = 0
    start = time.time()
    for e in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            test_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model.forward(inputs)
                    test_loss += criterion(log_ps, labels).item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e + 1}/{epochs}.. "
                      f"Train loss: {running_loss / len(trainloader):.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()

    print(f"Time per batch: {(time.time() - start) / 3:.3f} seconds")


def model_testing(model, trainloader, testloader, criterion, optimizer, device=in_arg.gpu):
    """
    The testing function - shows how well the trained model performs on a unknown data set - testing data set
    :param model: passing the already trained model
    :param trainloader: training data set
    :param testloader: testing data set
    :param criterion: the loss criterion
    :param optimizer: the backpropagation optimizer
    :param device: the device to be used in the testing process (gpu or cpu)
    """
    steps = 0
    running_loss = 0
    print_every = 5

    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad

        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model.forward(inputs)
                    test_loss += criterion(log_ps, labels).item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Steps {steps}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()


def predict(image_path, model, device=in_arg.gpu,  topk=in_arg.top_k):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    :param image_path: path to the image to be predicted
    :param model: passing the already trained model
    :param device: the device to be used in the prediction process (gpu or cpu)
    :param topk: the number of top classes (classes with biggest probability) to which the image belongs to
    """

    model.eval()

    with torch.no_grad():
        image = process_image(image_path)
        image = torch.FloatTensor([image])
        image, model = image.to(device), model.to(device)

        log_ps = model.forward(image)

        ps = torch.exp(log_ps)
        top_prob, top_class = ps.topk(topk, dim=-1)

        prob = top_prob.tolist()[0]  # probabilities
        index = top_class.tolist()[0]  # index

        ind = []
        for i in range(len(model.class_to_idx.items())):
            ind.append(list(model.class_to_idx.items())[i][0])

        # transfer index to label
        label = []
        for i in range(topk):
            label.append(ind[index[i]])

    return prob, label

