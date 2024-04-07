from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import args


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        out = self.activate(self.linear(x))
        return out


class Linear_Model:
    def __init__(self):
        self.learning_rate = 5e-4
        self.epoches = 100000
        self.loss_function = torch.nn.MSELoss()
        self.create_model()

    def create_model(self):
        self.model = LinearRegression().to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, num, data):

        x = data.unsqueeze(1)
        x = (x - torch.mean(x)) / torch.std(x)
        y = torch.arange(1, len(x) + 1).unsqueeze(1) / len(x)

        temp = 100
        for epoch in range(self.epoches):
            prediction = self.model(x.to(args.device))
            loss = self.loss_function(prediction, y.to(args.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10000 == 0:
                print("Epoch: {}  loss: {}".format(epoch + 1, loss.item()))
                if (temp - loss) < 1e-7:
                    break
                else:
                    temp = loss
        torch.save(self.model.state_dict(), "probs/linear_{}_{}.pth".format(num, args.dataset))

    def test(self, num, data):
        x = data.unsqueeze(1)
        x = (x - torch.mean(x)) / torch.std(x)

        self.model.load_state_dict(torch.load("probs/linear_{}_{}.pth".format(num, args.dataset)))
        self.model.eval()
        prediction = self.model(x).cpu()

        return (prediction * len(x)).to(dtype=torch.int32).squeeze(1).detach().numpy()


if __name__ == '__main__':
    pool = Pool(processes=10)
    for num in range(args.n_cluster):
        data = np.load('probs/probs_{}_{}.npy'.format(num, args.dataset))
        data = torch.Tensor(data)
        linear = Linear_Model()
        pool.apply_async(linear.train, (num, data))
    pool.close()
    pool.join()
