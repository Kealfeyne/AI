import torch
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

# Turn random off
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True

x_train = torch.tensor(pd.read_csv("train.csv", dtype=np.float32).values)
y_train = torch.tensor(pd.read_csv("trainTargets.csv", dtype=np.float32).values)
x_validation = torch.tensor(pd.read_csv("validation.csv", dtype=np.float32).values)
y_validation = torch.tensor(pd.read_csv("validationTargets.csv", dtype=np.float32).values)
x_test = torch.tensor(pd.read_csv("test.csv", dtype=np.float32).values)
y_test = torch.tensor(pd.read_csv("testTargets.csv", dtype=np.float32).values)


class Net(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(81, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


def learn(number_of_epochs, number_of_neurons):
    net = Net(number_of_neurons)

    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)

    batch_size = 110

    epochs_accuracy_history = []
    epochs_loss_history = []

    count_of_epochs = 0
    for epoch in range(number_of_epochs):
        order = np.random.permutation(len(x_train))

        for start_index in range(0, len(x_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            x_batch = x_train[batch_indexes]
            y_batch = y_train[batch_indexes]

            predictions = net.forward(x_batch)
            loss_value = loss(predictions, y_batch)
            loss_value.backward()

            optimizer.step()

        epoch_predictions = net.forward(x_validation)
        epochs_loss_history.append(loss(epoch_predictions, y_validation))
        accuracy = (abs(1 - abs(epoch_predictions - y_validation)) * 100.0).float().mean()
        epochs_accuracy_history.append(accuracy)
        count_of_epochs += 1
        print("Точность на ", count_of_epochs, " эпохе равна ", accuracy.item())


    # plt.xlabel("Epoch number")
    # plt.ylabel("Validation loss value")
    # plt.plot(epochs_loss_history, lw=2, label="s_butch=100")
    # plt.show()
    #
    # plt.xlabel("Epoch number")
    # plt.ylabel("Validation accuracy value")
    # plt.plot(epochs_accuracy_history, lw=2, label="s_butch=100")
    # plt.show()

    #return net # Если нужно проверить на Тестовом Дата Сете
    return epochs_accuracy_history # Если хотим провести эксперимент для множества значений количества нейронов


def test(net):
    test_predictions = net.forward(x_test)
    accuracy = (abs(1 - abs(test_predictions - y_test)) * 100.0).float().mean()
    print("Точность на тестовом Дата Сете равна ", accuracy.item())
    return accuracy.item()


def analysis(number_of_epochs, number_of_neurons):
    # Make data.
    neurons_accuracy_history = np.empty((number_of_neurons, number_of_epochs), dtype=np.float32)
    count_of_neurons = 1
    for neurons_version in range(number_of_neurons):
        print("Обучается сеть с количеством нейронов: ", count_of_neurons)
        accuracy_of_count_of_neurons = learn(number_of_epochs, number_of_neurons)
        neurons_accuracy_history[count_of_neurons-1] = accuracy_of_count_of_neurons
        count_of_neurons += 1

    X = np.linspace(1, number_of_epochs, number_of_epochs, dtype=np.float32)
    Y = np.linspace(1, number_of_neurons, number_of_neurons, dtype=np.float32)
    X, Y = np.meshgrid(X, Y)
    Z = neurons_accuracy_history

    # Plot figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 100,  cmap='PiYG')

    # Customize
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Number of neurons')
    ax.set_zlabel('Accuracy')

    plt.show()


# Main
number_of_epochs = 100
number_of_neurons = 150

#learn(number_of_epochs, number_of_neurons)
#test(learn()) # Если хочется проверить на тестовом Дата Сете (Также придется поменять return у функции learn)
analysis(number_of_epochs, number_of_neurons)