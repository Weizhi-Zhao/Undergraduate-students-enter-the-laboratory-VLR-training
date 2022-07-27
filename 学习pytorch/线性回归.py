import torch
import numpy as np
import torch.nn as nn

x_values = [i for i in range(100)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
#print(x_train.shape)

y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

class LinearRegressModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1

model = LinearRegressModel(input_dim, output_dim)

#model.load_state_dict(torch.load('LinearRegressModel.pkl'))

epochs = 100000
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

for epoch in range(1, epochs + 1):
    inputs = torch.from_numpy(x_train)
    lables = torch.from_numpy(y_train)

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, lables)

    loss.backward()

    optimizer.step()

    if epoch % 5000 == 0:
        print('epoch : {}  loss : {}'.format(epoch, loss.item()))
    # if epoch % 40000 == 0:
    #     for param_groups in optimizer.param_groups:
    #         param_groups['lr'] = learning_rate * (0.1 ** (epoch // 40000))
    #     print('learning rate :', learning_rate * (0.1 ** (epoch // 40000)))

#model.eval()

test_values = [i + 0.2 for i in range(100)]
test_inputs = np.array(test_values, dtype=np.float32)
test_inputs = test_inputs.reshape(-1, 1)

inputs = torch.from_numpy(test_inputs)
print(model(inputs))

torch.save(model.state_dict(), 'LinearRegressModel.pkl')