import torch
from torch import nn
import torch.nn.functional as F
import train_args
import time
from collections import OrderedDict

in_arg = train_args.args_input()


def classifier(input_size, output_size=102, hidden_layers_=in_arg.hidden_units, drop_p=in_arg.drop_out):
    hidden_layers = []
    for s in range(len(hidden_layers_)):
        hidden_layers.append(int(hidden_layers_[s]))
    sequetial_tuple = []
    sequetial_tuple.append((str(0), nn.Linear(input_size, hidden_layers[0])))
    sequetial_tuple.append((str(1), nn.Dropout(drop_p)))
    sequetial_tuple.append((str(2), nn.ReLU()))

    if len(hidden_layers) == 1:
        sequetial_tuple.append((str(3), nn.Linear(hidden_layers[0], output_size)))
        sequetial_tuple.append((str(4), nn.LogSoftmax(dim=1)))
    elif len(hidden_layers) >= 2:
        for n in range(len(hidden_layers) -1):
            sequetial_tuple.append((str(int(n+3)), nn.Linear(hidden_layers[n], hidden_layers[n + 1])))
            sequetial_tuple.append((str(int(n+4)), nn.Dropout(drop_p)))
            sequetial_tuple.append((str(int(n+5)), nn.ReLU()))

        sequetial_tuple.append((str(int(len(sequetial_tuple) +1)), nn.Linear(hidden_layers[len(hidden_layers) - 1], output_size)))
        sequetial_tuple.append((str(int(len(sequetial_tuple) +2)), nn.LogSoftmax(dim=1)))

    return nn.Sequential(OrderedDict(sequetial_tuple))


def train(model, trainloader, testloader, criterion, optimizer, epochs=in_arg.epochs, device=in_arg.gpu):
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
