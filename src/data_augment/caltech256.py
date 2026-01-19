import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .utils import DefaultDataset, DataAugmentationDataset


num_workers = 16
batch_size = 128
num_epochs = 5
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(aug_method):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    caltech256_dataset = datasets.Caltech256(root='data/source', download=True, transform=lambda x: x.convert('RGB') if x.mode != 'RGB' else x)
    X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
    y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=18, stratify=y)
    trainset = DefaultDataset(X_train, y_train)
    testset = DefaultDataset(X_test, y_test, transform=transform)
    if aug_method is None or aug_method == 'no':
        trainset.transform = transform
        return trainset, testset
    augmented_trainset = DataAugmentationDataset(
        dataname='Caltech256',
        dataset=trainset,
        get_img_from_item_fn=lambda item: item[0],
        prepare_item_fn=lambda img, item: (img, item[1]),
        aug_method=aug_method,
        transform=transform
    )
    return augmented_trainset, testset


def run(aug_method):
    os.makedirs('models', exist_ok=True)

    trainset, testset = load_dataset(aug_method=aug_method)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # train
    model = models.densenet121(weights='DEFAULT')
    num_classes = 257
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model_path = f'models/Caltech256_DenseNet121_Aug_{aug_method}_tmp.pth'
        torch.save(model.state_dict(), model_path)

    # test
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    model_path = f'models/Caltech256_DenseNet121_Aug_{aug_method}_{int(accuracy*100)}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
