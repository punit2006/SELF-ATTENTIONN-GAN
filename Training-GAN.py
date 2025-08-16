def train_gan(dataloader, epochs=50):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(tqdm(dataloader)):
            # Move images to device
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Forward pass real images
            outputs = torch.sigmoid(discriminator(real_images))
            d_loss_real = criterion(outputs, real_labels)

            # Generate fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            outputs = torch.sigmoid(discriminator(fake_images.detach()))
            d_loss_fake = criterion(outputs, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            outputs = torch.sigmoid(discriminator(fake_images))
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

        # Log to wandb
        wandb.log({"Generator Loss": g_loss.item(), "Discriminator Loss": d_loss.item()})
