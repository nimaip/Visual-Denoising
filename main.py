import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time

# --- Definitions can stay in the global scope ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)
        return image

class JInv_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(JInv_UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.final_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.upconv1(x3)
        x5 = torch.cat([x4, x1], dim=1)
        x6 = self.dec1(x5)
        
        w = self.final_conv1.weight.clone()
        center_h, center_w = w.shape[2] // 2, w.shape[3] // 2
        w[:, :, center_h, center_w] = 0
        x7 = nn.functional.conv2d(x6, w, self.final_conv1.bias, padding=1)
        x7 = nn.functional.relu(x7)
        out = self.final_conv2(x7)
        return out

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),      
    ])

    image_dir = "C:/users/Nimai/Documents/Vis_Denoise/images"
    batch_size = 4   
    learning_rate = 0.001
    num_epochs = 10   

    if not os.path.isdir(image_dir) or not any(f.lower().endswith('.tif') for f in os.listdir(image_dir)):
        print(f"Error: Directory '{image_dir}' not found or contains no .tif files.")
        print("Please update the 'image_dir' variable to point to your dataset folder.")
    else:
        train_dataset = MedicalImageDataset(image_dir=image_dir, transform=transform)
        # Setting num_workers > 0 requires the if __name__ == '__main__': guard
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        print(f"Successfully loaded {len(train_dataset)} images.")

        model = JInv_UNet().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print("\nStarting model training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for batch_images in train_loader:
                batch_images = batch_images.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_images)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_images.size(0)
                
            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
            
        end_time = time.time()
        print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")

        print("\nVisualizing results on a sample batch...")
        model.eval()
        with torch.no_grad():
            sample_images = next(iter(train_loader)).to(device)
            
            denoised_images = model(sample_images)

            sample_images = sample_images.cpu().numpy()
            denoised_images = denoised_images.cpu().numpy()

            n_images_to_show = min(batch_size, 4)
            fig, axes = plt.subplots(2, n_images_to_show, figsize=(15, 8))
            for i in range(n_images_to_show):
                
                axes[0, i].imshow(np.squeeze(sample_images[i]), cmap='gray')
                axes[0, i].set_title("Original Noisy")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(np.squeeze(denoised_images[i]), cmap='gray')
                axes[1, i].set_title("Denoised by N2I")
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()