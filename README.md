# Self-Attention GAN (PyTorch)

This project implements a **Self-Attention Generative Adversarial Network (SAGAN)** using PyTorch.
The model is trained on the **CIFAR-10 dataset** and can generate **64×64 RGB images**.

## ✨ Features

* Self-Attention mechanism in both **Generator** and **Discriminator**
* Trains on **CIFAR-10** dataset (resized to 64×64)
* Uses **WandB** for experiment tracking
* Saves and reloads trained **Generator** & **Discriminator**
* Generates and saves sample images

---

## 📦 Requirements

Install the required libraries (auto-installed in Colab):

```bash
pip install torch torchvision matplotlib numpy pandas tqdm wandb
```

---

## 🚀 Usage

### 1. Run in Google Colab

You can run the full project in Colab here:

👉 https://colab.research.google.com/drive/1bKwvxt9WDf24eERPtgxQEHdEFjPj6C54?usp=sharing

---

### 2. Track Training in WandB

This project integrates with **Weights & Biases** for experiment tracking.

👉 https://wandb.ai/punit163-work-student/attention-gan/runs/75y2mu19/workspace?nw=nwuserpunit163work

Example init:

```python
import wandb
wandb.init(project="attention-gan")
```

If you don’t want logging, disable it with:

```python
wandb.init(mode="disabled")
```

---

### 3. Train the GAN

Run the script locally or in Colab:

```bash
python self_attentionn_gan.py
```

This will:

* Train the GAN for 50 epochs
* Log losses to WandB
* Save trained models as:

  * `generator.pth`
  * `discriminator.pth`

---

### 4. Generate Images

After training:

```python
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from self_attentionn_gan import Generator
generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

noise = torch.randn(16, latent_dim, 1, 1, device=device)
with torch.no_grad():
    fake_images = generator(noise).cpu()

vutils.save_image(fake_images, "generated_samples.png", normalize=True, nrow=4)
```

---

### 5. Example Output

Generated images will be saved as **`generated_samples.png`**:

```
generated_samples.png
```

---

## 📂 Project Structure

```
self_attentionn_gan.py   # Main training + generation script
generator.pth            # Saved Generator model
discriminator.pth        # Saved Discriminator model
generated_samples/       # Folder for generated images
```

---

## 📌 Notes

* Replace the **WandB link** with your actual project dashboard link after the first run.
* Training on **Colab GPU (T4)** takes \~30-40 minutes for 50 epochs.

