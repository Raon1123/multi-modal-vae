import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# finetune the model on cifar 10

def finetune(model, train_loader, optimizer, epoch, device, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       loss.item()))


def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # calculate accuracy
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)

    return correct / total
        

def main():
    # Training settings
    batch_size = 128
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # pretrained resnet18
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(512, 10)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    acc_list = []

    for epoch in range(1, epochs + 1):
        finetune(model, train_dataloader, optimizer, epoch, device)
        acc = test_model(model, test_dataloader, device)
        acc_list.append(acc)

    # save model
    torch.save(model.state_dict(), 'pretrained_model.pt')

    # print accuracy
    print(acc_list)
    


if __name__ == '__main__':
    main()
