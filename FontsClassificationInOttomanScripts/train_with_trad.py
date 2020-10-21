import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda, device, optim
from torch.cuda import is_available
from torchvision import datasets, models, transforms
from tqdm import tqdm



device = torch.device("cuda:0" if is_available() else "cpu")

IMG_W = 500
IMG_H = 50
NUM_CLASSES = 2
training_data = np.load("training_data.npy", allow_pickle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1*IMG_W*IMG_H, out_features=256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x


net = Net().to(device=device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


X = torch.Tensor([i[0] for i in training_data]).view(-1, IMG_W, IMG_H)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size].cuda()
train_y = y[:-val_size].cuda()

test_X = X[-val_size:]
test_y = y[-val_size:]

BATCH_SIZE = 5
EPOCHS = 10


def train(net):
    loss = ""
    for epoch in range(EPOCHS):
        # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            # print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, IMG_W, IMG_H)
            # print(batch_X.shape)
            batch_y = train_y[i:i+BATCH_SIZE]

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")


model_path = "checkpoint.pth"


def save(net):
    """
    Save my model
    """
    # checkpoint = {
    #     "input_size":1*IMG_W*IMG_H,
    #     "output_size":NUM_CLASSES,
    #     "hidden_layers":[each.out_features for each in net.hidden_layers],
    #     "state_dict": net.state_dict()
    # }

    torch.save(net.state_dict(), model_path)
    # torch.save(checkpoint, model_path)


def load(path):
    """
    Load my Model
    """
    # checkpoint = torch.load(path)
    # model = fc_model.Network(checkpoint["input_size"], checkpoint["output_size"],checkpoint["hidden_layers"])

    # model.load_state_dict(checkpoint["state_dict"])

    model = Net().to(device=device)
    model.load_state_dict(torch.load(path))
    return model



net =  Net().to(device=device)
train(net)
save(net)



test_X.to(device)
test_y.to(device)


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            # returns a list,
            net_out = net(test_X[i].view(-1, 1, IMG_W, IMG_H).to(device))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))


net = load(model_path)
test(net)
