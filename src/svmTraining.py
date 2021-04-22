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


