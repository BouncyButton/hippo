import monai
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, CropForegroundd, RandCropByPosNegLabeld, ToTensord
from monai.data import Dataset, DataLoader
import torch
import os

# -----------------------------
# 1. Define transforms
# -----------------------------
train_transforms = Compose([
    LoadImaged(keys=['image','label']),
    EnsureChannelFirstd(keys=["image","label"], channel_dim="no_channel"),
    ScaleIntensityd(keys=['image']),
    RandCropByPosNegLabeld(keys=['image','label'], label_key='label', spatial_size=(64,64,64), pos=1, neg=1, num_samples=4),
    ToTensord(keys=['image','label'])
])

val_transforms = Compose([
    LoadImaged(keys=['image','label']),
    EnsureChannelFirstd(keys=["image","label"], channel_dim="no_channel"),
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image','label'])
])

# -----------------------------
# 2. Prepare dataset
# -----------------------------
# Example: [{"image": "subj001.nii.gz", "label": "subj001_label.nii.gz"}, ...]
train_files = [...]
val_files = [...]

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# -----------------------------
# 3. Define model
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,  # hippocampus vs background
    channels=(16,32,64,128,256),
    strides=(2,2,2,2),
    num_res_units=2
).to(device)

loss_function = monai.losses.DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# 4. Training loop (simplified)
# -----------------------------
max_epochs = 100

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    for batch_data in train_loader:
        inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss/len(train_loader)}")
