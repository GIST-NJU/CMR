import numpy as np
from scipy.stats import gamma
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2 )
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent_params = self.encoder(x)
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

def train_vae():
    save_path = 'results/validity/MNIST_VAE.pth'
    if os.path.exists(save_path):
        model.load_state_dict(torch.load('results/validity/MNIST_VAE.pth'))
        return

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='sum')
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_size).to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            BCE = criterion(recon_batch, data)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + KLD
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader.dataset)}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.view(-1, input_size).to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += criterion(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print(f"Test set loss: {test_loss:.4f}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def calculate_threshold():
    save_path = 'results/validity/MNIST_threshold.txt'
    if os.path.exists(save_path):
        return

    criterion = nn.BCELoss(reduction='mean')
    error_testing = np.zeros(len(test_set))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.view(-1, input_size).to(device)
            recon_batch, _, _ = model(data)
            error_testing[i] = criterion(recon_batch, data).item()

    shape, loc, scale = gamma.fit(error_testing, floc=0)
    false_alarm = 0.0001
    threshold = gamma.ppf(1-false_alarm, shape, loc, scale)
    # print(threshold)
    # print(np.where(error_testing>threshold)[0].size)
    with open(save_path, 'w') as f:
        f.write(f'False_alarm: {false_alarm}\n')
        f.write(f'Threshold: {threshold}\n')

def predict_validity():
    criterion = nn.BCELoss(reduction='mean')
    result_selfOracle = {}
    batch_size = 1
    model.eval()
    model.to(device)

    followup_dir = 'data/followup/MNIST'
    entries = os.listdir(followup_dir)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    for folder in folders:
        cmr = tuple(int(char) for char in folder)
        result_selfOracle[cmr] = []
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, transform=transform)
        followup_test_loader = DataLoader(followup_test_set, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(followup_test_loader):
                data = data.view(-1, input_size).to(device)
                recon, _, _ = model(data)
                error = criterion(recon, data)
                result_selfOracle[cmr].append(error.cpu().item())
        print(cmr)
    np.save('results/validity/MNIST_validity.npy', result_selfOracle)

def run():
    train_vae()
    calculate_threshold()
    predict_validity()

batch_size = 128
input_size = 28 * 28
hidden_size = 400
latent_size = 200
lr = 1e-3
epochs = 50
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set = datasets.MNIST('data/source', train=True, download=True, transform=transform)
test_set = datasets.MNIST('data/source', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_size, hidden_size, latent_size).to(device)