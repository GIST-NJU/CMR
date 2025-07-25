class CustomDataset(Dataset):
    def __init__(self, dataset, cmr=None, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for idx, (img, label) in enumerate(dataset):
            if cmr is not None:
                for index in cmr:
                    img = mrs[index](img, paras[index])
            self.data.append(img)
            self.labels.append(label)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    
# 设置随机种子
torch.manual_seed(18)

#transform = models.DenseNet121_Weights.IMAGENET1K_V1.transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3), # the mode of some images is L not RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 Caltech-256 数据集
caltech256_dataset = datasets.Caltech256(root='data', download=True)

 
X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]

# X = []
# y = []
# for i in range(len(caltech256_dataset)):
#     if caltech256_dataset[i][1]!=256:
#         X.append(caltech256_dataset[i][0])
#         y.append(caltech256_dataset[i][1])
# print(len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=18, stratify=y)

train_set = CustomDataset(zip(X_train, y_train), transform=transform)
test_set = CustomDataset(zip(X_test, y_test), transform=transform)

# for i in range(len(train_set)):
#     print(type(train_set[i][0]))

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 构建 DenseNet 模型
densenet = models.densenet121(weights='DEFAULT')  # 使用预训练的 DenseNet-121 模型
# 修改最后的全连接层以适应 Caltech-256 数据集的输出类别数量
num_classes = 257  # 注意类别数量可能需要根据实际情况进行调整
densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)



# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)

# 模型训练
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet.to(device)
print(device)

densenet.train()  # 设置模型为训练模式
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader)}")

# 测试模型
densenet.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = densenet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(100 * correct / total):.2f}%")

# 设置随机种子
torch.manual_seed(18)

#transform = models.DenseNet121_Weights.IMAGENET1K_V1.transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3), # the mode of some images is L not RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 Caltech-256 数据集
caltech256_dataset = datasets.Caltech256(root='data', download=True)

 
X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]

# X = []
# y = []
# for i in range(len(caltech256_dataset)):
#     if caltech256_dataset[i][1]!=256:
#         X.append(caltech256_dataset[i][0])
#         y.append(caltech256_dataset[i][1])
# print(len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=18, stratify=y)

train_set = CustomDataset(zip(X_train, y_train), transform=transform)
test_set = CustomDataset(zip(X_test, y_test), transform=transform)

# for i in range(len(train_set)):
#     print(type(train_set[i][0]))

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 构建 DenseNet 模型
densenet = models.densenet121(weights='DEFAULT')  # 使用预训练的 DenseNet-121 模型
# 修改最后的全连接层以适应 Caltech-256 数据集的输出类别数量
num_classes = 257  # 注意类别数量可能需要根据实际情况进行调整
densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)



# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)

# 模型训练
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet.to(device)
print(device)

densenet.train()  # 设置模型为训练模式
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader)}")

# 测试模型
densenet.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = densenet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(100 * correct / total):.2f}%")

torch.manual_seed(18)

model_name = 'DenseNet121_6838'
model = models.densenet121()
num_classes = 257  # 注意类别数量可能需要根据实际情况进行调整
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load('./models/'+model_name+'.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 1024

# 数据集根目录和预处理
root_dir = '/path/to/caltech256'
#transform = models.DenseNet121_Weights.IMAGENET1K_V1.transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3), # the mode of some images is L not RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 Caltech-256 数据集
caltech256_dataset = datasets.Caltech256(root='data', download=True)

 
X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=18, stratify=y)
test_set = CustomDataset(zip(X_test, y_test))

pred_followup = {}
for k in range(len(mrs)): 
    k = k+1
    for p in permutations(range(len(mrs)), k):
        if p in pred_followup.keys():
            continue
        print(p)
        temp = np.zeros(len(test_set), dtype=int)
        dateset_followup = CustomDataset(test_set, cmr=p, transform=transform)
        testload_followup = torch.utils.data.DataLoader(dateset_followup, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for i,(X,y) in enumerate(testload_followup):
                X = X.to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                temp[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
        pred_followup[p] = temp