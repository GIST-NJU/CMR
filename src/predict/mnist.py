mnist_testset = datasets.MNIST(root='./data', train=False, download=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小以适应预训练模型的输入大小
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为三通道输入
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model_name = 'MNIST_ResNet18_9906'
model.load_state_dict(torch.load('./models/'+model_name+'.pth'))
model.eval()
model.to(device)

correct = 0
total = 0
pred_source = np.zeros(len(mnist_testset),dtype=int)
#pred_source = np.load('predictions/MNIST_ResNet18_9906_source.npy')
with torch.no_grad():
    for i,(X,y) in enumerate(test_loader):
        X,y = X.to(device),y.to(device)
        outputs = model(X)
        _, pred = torch.max(outputs, 1)
        pred_source[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
        correct += (pred==y).sum().item()
        total += y.size(0)

accuracy = correct/total
print(accuracy)


class MNISTFOLLOWUP(Dataset):
    def __init__(self, mnist_dataset, cmr, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for idx, (img, label) in enumerate(mnist_dataset):
            for index in cmr:
                img = mrs[index](img, paras[index])
            self.data.append(img)
            self.labels.append(label)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            self.data[idx] = self.transform(self.data[idx])
        return self.data[idx], self.labels[idx]
    

pred_followup = {}
for k in range(len(mrs)): 
    k = k+1
    for p in permutations(range(len(mrs)), k):
        if p in pred_followup.keys():
            continue
        print(p)
        temp = np.zeros(len(mnist_testset), dtype=int)
        dateset_followup = MNISTFOLLOWUP(mnist_dataset=mnist_testset, cmr=p, transform=transform)
        testload_followup = torch.utils.data.DataLoader(dateset_followup, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for i,(X,y) in enumerate(testload_followup):
                X = X.to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                temp[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
        pred_followup[p] = temp