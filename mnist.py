import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPNet(nn.Module):
    """原来的全连接网络版本（保留作为参考）。
    输出 logits（不再手动 log_softmax），配合 CrossEntropyLoss 使用。
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class CNNNet(nn.Module):
    """更适合 MNIST 的卷积网络。两层卷积 + 池化 + 两层全连接。
    在 MNIST 上 1~2 个 epoch 就能达到 99% 左右。
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # [N,32,14,14]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # [N,64, 7, 7]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def get_data_loader(is_train, batch_size=128):
    # MNIST 全数据集统计值，做标准化能加快收敛
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_set = MNIST(root='mnist_data/', train=is_train,
                     download=True, transform=transform)
    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=(device.type == 'cuda'),
        num_workers=0,
    )


@torch.no_grad()
def evaluate(test_data, net):
    net.eval()
    n_correct, n_total = 0, 0
    for x, y in test_data:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        outputs = net(x)
        # 向量化：直接对整个 batch 取 argmax 与标签比较
        preds = outputs.argmax(dim=1)
        n_correct += (preds == y).sum().item()
        n_total += y.size(0)
    return n_correct / n_total


def train_one_epoch(net, train_data, optimizer, criterion, epoch):
    net.train()
    running_loss, seen = 0.0, 0
    for x, y in train_data:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        seen += y.size(0)
    return running_loss / seen


def visualize_predictions(net, test_data, n_show=8):
    net.eval()
    images, labels = next(iter(test_data))
    images_dev = images.to(device)
    with torch.no_grad():
        preds = net(images_dev).argmax(dim=1).cpu()

    n_show = min(n_show, images.size(0))
    cols = 4
    rows = (n_show + cols - 1) // cols
    plt.figure(figsize=(2.2 * cols, 2.4 * rows))
    for i in range(n_show):
        plt.subplot(rows, cols, i + 1)
        # 反标准化只为了看得更清楚
        img = images[i].squeeze().numpy() * 0.3081 + 0.1307
        plt.imshow(img, cmap='gray')
        color = 'green' if preds[i].item() == labels[i].item() else 'red'
        plt.title(f"pred:{preds[i].item()} / gt:{labels[i].item()}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    train_data = get_data_loader(is_train=True, batch_size=128)
    test_data = get_data_loader(is_train=False, batch_size=512)

    # 想用 MLP 就改成 MLPNet()
    net = CNNNet().to(device)
    print(net)
    print(f"using device: {device}")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    print("initial accuracy:", evaluate(test_data, net))

    epochs = 3
    for epoch in range(epochs):
        loss = train_one_epoch(net, train_data, optimizer, criterion, epoch)
        acc = evaluate(test_data, net)
        scheduler.step()
        print(f"epoch:{epoch}  loss:{loss:.4f}  test_acc:{acc:.4f}")

    visualize_predictions(net, test_data)


if __name__ == '__main__':
    main()
