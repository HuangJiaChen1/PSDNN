import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
LATENT_DIM = 32
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom dataset class to load images and resize them
class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to 64x64
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image


# Encoder network
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)

        x = x.view(x.size(0), -1)  # Flatten

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


# Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape

        x = F.leaky_relu(self.deconv1(x), 0.2)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = torch.tanh(self.deconv4(x))  # Output values between -1 and 1

        return x


# VAE class
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def generate(self, num_samples):
        """Generate new images from the latent space"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(DEVICE)
            samples = self.decoder(z)
        return samples


# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (binary cross entropy)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


# Training function
def train(model, dataloader, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(dataloader.dataset)


# Function to save generated images
def save_generated_images(model, epoch):
    model.eval()
    with torch.no_grad():
        sample = model.generate(10)
        sample = sample.cpu()

        # Denormalize images
        sample = sample * 0.5 + 0.5

        plt.figure(figsize=(10, 4))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(sample[i].permute(1, 2, 0).numpy())
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'generated_images_epoch_{epoch}.png')
        plt.close()


# Function to save individual images as JPG files
def save_individual_images(model, output_folder):
    """Save 10 generated images as individual JPG files in the specified folder"""
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Generate 10 images
        sample = model.generate(10)
        sample = sample.cpu()

        # Denormalize images
        sample = sample * 0.5 + 0.5

        # Save each image as a JPG file
        for i in range(10):
            # Convert from tensor to PIL Image
            img = sample[i].permute(1, 2, 0).numpy()
            # Clip values to [0, 1] range to avoid warnings
            img = np.clip(img, 0, 1)
            # Convert to PIL Image
            img = Image.fromarray((img * 255).astype('uint8'))
            # Save as JPG
            img.save(os.path.join(output_folder, f'generated_image_{i + 1}.jpg'))


# Main training loop
def main():
    # Load dataset
    dataset = ImageDataset('datasets/partA/examplars')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = VAE(LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, dataloader, optimizer)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

        # Save generated images every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            save_generated_images(model, epoch)

    # Save the trained model
    torch.save(model.state_dict(), 'vae_model.pth')

    # Save 10 individual JPG images in the EXAMPLARS folder
    save_individual_images(model, 'EXAMPLARS')

    print(f"10 generated images have been saved as individual JPG files in the 'EXAMPLARS' folder.")


if __name__ == "__main__":
    main()