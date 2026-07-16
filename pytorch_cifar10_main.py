"""
CIFAR10 training script (legacy single-file entry point, kept for compatibility).
Recommended: run_train.py (unified config, TensorBoard, results directory) and run_parallel.py (multi-config multi-GPU parallel).
"""
import torchvision.transforms as transforms
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd

# Import model
# from backbones.ProbResNet import ProbResNet18
from backbones.ProbGoogLeNet import ProbGoogLeNet

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# For measuring GPU runtime
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_acc = 0
    # Start iterating over each batch
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs:[b,3,32,32], targets:[b]
        # train_outputs:[b,10]
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        # print(targets.shape)
        labels_one_hot = torch.zeros(targets.size(0), 10, device=device)
        labels_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.shape)
        # print(labels_one_hot.shape)
        loss = criterion(outputs, labels_one_hot)
        loss.backward()
        optimizer.step()

        # Compute loss
        train_loss += loss.item()
        # print(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Compute accuracy
        train_acc = correct / total
        # Print train loss and accuracy every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                         batch_idx + 1,
                                                                                         loss.item(),
                                                                                         train_acc))
    # Compute training accuracy within each epoch
    total_train_acc.append(train_acc)


# Testing
def test(epoch, ckpt):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            labels_one_hot = torch.zeros(targets.size(0), 10, device=device)
            labels_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        print(
            '[INFO] Epoch-{}-Test Accuracy: {:.3f}'.format(epoch + 1, test_acc), '\n')

    total_test_acc.append(test_acc)

    # Save checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt)
        best_acc = acc


if __name__ == '__main__':
    # Set hyperparameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data', type=str, default='CIFAR10')
    parser.add_argument('--T_max', type=int, default=100)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/ResNet18-CIFAR10.pth')
    opt = parser.parse_args()

    # Set parameters
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    # print(device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Set data augmentation
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.1307,), (0.3081,))
        # transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.1307,), (0.3081,))
        # transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root=opt.data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=opt.data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Download and load MNIST digits training dataset
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    # Download and load MNIST digits test dataset
    # test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    # testloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # Download FashionMNIST training dataset
    # train_dataset = torchvision.datasets.FashionMNIST(root='./fashion', train=True, download=True,
    #                                                  transform=transform_train)
    # trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    # Download FashionMNIST test dataset
    # test_dataset = torchvision.datasets.FashionMNIST(root='./fashion', train=False, download=True,
    #                                                 transform=transform_test)
    # testloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    # Load model
    print('==> Building model..')
    # model = ProbResNet18().to(device)
    model = ProbGoogLeNet().to(device)
    # model = PbitNet3().to(device)

    # DataParallel training
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Load previously trained weights
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=5e-4)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    # Cosine annealing: LR smoothly decays from lr to near 0 per epoch (commonly used with AdamW)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # Track train and test accuracy
    total_test_acc = []
    total_train_acc = []

    # Record training time
    tic = time_sync()

    # Start training
    for epoch in range(opt.epochs):
        train(epoch)
        test(epoch, opt.checkpoint)
        scheduler.step()  # Update LR via cosine annealing after each epoch

    # Save train/test accuracy to Excel
    os.makedirs('output', exist_ok=True)
    acc_df = pd.DataFrame({
        'Epoch': range(opt.epochs),
        'Train_Acc': total_train_acc,
        'Test_Acc': total_test_acc,
    })
    acc_df.to_excel('output/ResNet18-CIFAR10-Accuracy.xlsx', index=False)

    # Data visualization
    plt.figure()
    plt.plot(range(opt.epochs), total_train_acc, label='Train Accuracy')
    plt.plot(range(opt.epochs), total_test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('ResNet18-CIFAR10-Accuracy')
    plt.legend()
    # Auto-save plotted image
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/ResNet18-CIFAR10-Accuracy.jpg')
    plt.show()

    # Output best_acc
    print(f'Best Acc: {best_acc}%')
    toc = time_sync()
    # Calculate total runtime
    t = (toc - tic) / 3600
    print(f'Training Done. ({t:.3f}s)')
