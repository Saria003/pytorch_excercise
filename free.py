import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from helper_functions import plot_predictions, plot_decision_boundary
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import requests
import math # YEAH! MATH BITCH!

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.layer_1(x)
        # x = self.relu(x)
        # x = self.layer_2(x)
        # x = self.relu(x)
        # x = self.layer_3(x)
        # return x
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

def accuracy_fn(y_true, y_pred):
    # res = (y_pred.round() == y).float().mean()
    y_true = y_true.to(y_pred.device) # just to make sure they are both on the same device
    correct = torch.eq(y_true, y_pred).sum().item()
    # try:
    acc = (correct / len(y_pred)) * 100
    # acc = (correct / len() * 100
    return acc
    # except ZeroDivisionError:
        # print("You are dividing by zero")
        # print(f"(correct) {correct} / (len of y_pred) {len(y_pred)}")

# Hyperparameters
lr = 0.01

# Creating non linear model
n_samples = 1000
torch.manual_seed(42)
torch.cuda.manual_seed(42)
X, y = make_circles(n_samples, 
                    noise=0.03, 
                    random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

# Split
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.2,
                                                    random_state=42)

    
model_3 = CircleModelV2()

# loss_fn & optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(params=model_3.parameters(), lr=lr)

# training loop
epochs = 100
batch_size = 20
# batches_per_epoch = len(X) // batch_size # 1000 // 50 = 20
# batches_per_epoch = math.ceil(len(X) // batch_size) # 1000 // 50 = 20
batches_per_epoch = math.ceil(len(X_train) // batch_size) # 800 // 20 = 40

for epoch in range(epochs):
    model_3.train()

    for i in range(batches_per_epoch):
        start = i * batch_size # 50, 100, 150, 200 .etc 
        end = (i + 1) * batch_size # 100, 150, 200, 250 .etc 
        # one batch: X[50: 100]
        X_batch, y_batch = X_train[start:end], y_train[start:end]
        # print("length of X_batch: \n",len(X_batch))
        # print("length of y_batch: \n",len(y_batch))

        y_logit = model_3(X_batch).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logit))
        # print("y_pred: ", y_pred)
        loss = loss_fn(y_logit, y_batch)
        ## acc = accuracy_fn(y_train, y_pred)
        
        acc = accuracy_fn(y_batch, y_pred)
        # print(f"Acc in training loop: {acc}")
        
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("____________________________________________________________________")
            print(f"Training:\n Epoch: {epoch} | step: {i} | loss: {loss} | acc: {acc}")
    
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(y_test ,test_logits)
        test_acc = accuracy_fn(y_test ,test_pred)
        if epoch % 10 == 0:
            print("____________________________________________________________________")
            print(f"Testing\nEnd of {epoch}, test accuracy {test_acc:.4f}, test loss {test_loss:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
# plt.show()

plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
# plt.show()
