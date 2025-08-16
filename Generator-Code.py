class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_map=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: [batch_size, latent_dim, 1, 1]
            # Output: [batch_size, feature_map * 8, 4, 4]
            nn.ConvTranspose2d(latent_dim, feature_map * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),

            # Add self-attention
            SelfAttention(feature_map * 8),

            # Output: [batch_size, feature_map * 4, 8, 8]
            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(True),

            # Output: [batch_size, feature_map * 2, 16, 16]
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(True),

            # Output: [batch_size, feature_map, 32, 32]
            nn.ConvTranspose2d(feature_map * 2, feature_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),

            # Output: [batch_size, img_channels, 64, 64]
            nn.ConvTranspose2d(feature_map, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        return self.main(x)
