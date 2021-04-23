import torch 
import torch.optim as optim
from torch.autograd import Variable

from svmModel import SVM

def trainSVM(X, Y):
    learning_rate = 0.1
    epochs = 10
    batch_size = 1
    c = 0.01

    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    n = len(Y)

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
