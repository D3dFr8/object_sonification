import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class PointToIrNN(nn.Module):
    def __init__(self, input_dim=2,ir_length=512,hidden_sizes=[128,128,64,32]):
        super().__init__()

        layers = []
        last_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size,h))
            layers.append(nn.ReLU())
            last_size = h

        layers.append(nn.Linear(last_size, 2*ir_length))

        self.network = nn.Sequential(*layers)
        self.ir_length = ir_length

    def forward(self, x):
        out = self.network(x)
        return out.view(-1,2,self.ir_length)
    

def trainNetwork(net, sourcePositions, hrir, num_epochs=100, batch_size=16, lr=1e-3):
    X = torch.tensor(sourcePositions,dtype=torch.float32)
    Y = torch.tensor(hrir,dtype=torch.float32)

    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)

    X_norm = (X - X_mean) / X_std

    dataset = TensorDataset(X_norm, Y)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for xb,yb in dataloader:
            optimizer.zero_grad()
            pred = net(xb)
            loss = loss_fcn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)

    return net, X_mean, X_std