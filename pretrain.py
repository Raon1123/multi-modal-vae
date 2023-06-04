import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse

# finetune the model

def parse_args():
    parser = argparse.ArgumentParser(description='Train classifier')
    # general
    parser.add_argument('--dataset',
                        help="dataset",
                        default='cifar10',
                        type=str)

    parser.add_argument('--scheduler',
                        help="scheduler",
                        default=1,
                        type=int)
    args = parser.parse_args()

    return args

def finetune(model, train_loader, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


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
    args = parse_args()
    # Training settings
    batch_size = 128
    epochs = 30
    lr = 0.001
    momentum = 0.5
    seed = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset_name = args.dataset
    dataset_class = None
    if dataset_name == 'cifar10':
        dataset_class = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset_class = datasets.CIFAR100
    elif dataset_name == 'mnist':
        dataset_class = datasets.MNIST
    train_dataset = dataset_class('./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))
                                        ]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_dataset = dataset_class('./data', train=False, transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))
                                        ]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # pretrained resnet18
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    if dataset_name == 'cifar100':
        model.fc = nn.Linear(512, 100)
    else:
        model.fc = nn.Linear(512, 10)
    if dataset_name == 'mnist':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if args.scheduler:
        scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)

    acc_list = []

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        finetune(model, train_dataloader, optimizer, epoch, device)
        acc = test_model(model, test_dataloader, device)
        acc_list.append(acc)
        if args.scheduler:
            scheduler.step()
        pbar.set_description(f'Epoch: {epoch}, Accuracy: {acc}')

    # save model
    torch.save(model.state_dict(), f'resnet18_{dataset_name}_pretrained.pt')

    # print accuracy
    print(acc_list)
    


if __name__ == '__main__':
    main()
