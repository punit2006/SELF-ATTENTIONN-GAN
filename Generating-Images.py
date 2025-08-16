import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

os.makedirs("generated_samples", exist_ok=True)

num_images = 16
noise = torch.randn(num_images, latent_dim, 1, 1, device=device)

with torch.no_grad():
    fake_images = generator(noise).cpu()

vutils.save_image(fake_images, "generated_samples/fake_samples.png", normalize=True, nrow=4)

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True), (1,2,0)))
plt.show()
