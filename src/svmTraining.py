import torch 
import torch.optim as optim
from torch.autograd import Variable

from svmModel import SVM
from sklearn.model_selection import train_test_split

def trainClassifier(X, Y, val_split=0.20, num_classes=20):
    train_x, train_y = [], []
    val_x, val_y = [], []

    train_x, val_x, train_y, val_y = train_test_split(X, Y, stratify=Y, test_size=val_split)

    trainSVM(train_x, train_y, val_x, val_y)
    return

def trainSVM(X, Y, val_X, val_Y):
    learning_rate = 0.1
    epochs = 10
    batch_size = 1
    c = 0.01

    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    n = len(Y)

    val_X = torch.FloatTensor(val_X)
    val_Y = torch.FloatTensor(val_Y)
    val_n = len(val_Y)

    model = SVM()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        sum_loss = 0
        perm = torch.randperm(n)

        for i in range(0, n, batch_size):
            x = X[perm[i: i + batch_size]]
            y = Y[perm[i: i + batch_size]]

            x = Variable(x)
            y = Variable(y)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - output * y, min = 0))
            loss += c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {}, Loss: {}".format(epoch, sum_loss / n))
