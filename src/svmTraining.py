import torch 
import torch.optim as optim
from torch.autograd import Variable

from svmModel import SVM

learning_rate = 0.1
epochs = 10
batch_size = 1

X = torch.FloatTensor(X)
Y = torch.FloatTensor(Y)
n = len(y)

model = SVM()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

model.train()

for epoch in range(epochs):
    sum_loss = 0

    for i in range(0, n, batch_size):
        x = X
        y = Y

        x = Variable(x)
        y = Variable(y)

        optimizer.zero_grad()
        output = model(x)

        loss = torch.mean(torch.clamp(1 - output * y, min = 0))
        loss.backward()
        optimizer.step()

        sum_loss += loss[0].data.cpu().numpy()

    print("Epoch: {}, Loss: {}".format(epoch, sum_loss[0]))
