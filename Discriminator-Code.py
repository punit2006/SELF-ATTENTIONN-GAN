class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_map=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: [batch_size, img_channels, 64, 64]
            # Output: [batch_size, feature_map, 32, 32]
            nn.Conv2d(img_channels, feature_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Add self-attention
            SelfAttention(feature_map),

            # Output: [batch_size, feature_map * 2, 16, 16]
            nn.Conv2d(feature_map, feature_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: [batch_size, feature_map * 4, 8, 8]
            nn.Conv2d(feature_map * 2, feature_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: [batch_size, 1, 4, 4]
            nn.Conv2d(feature_map * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        x = self.main(x)
        x = torch.mean(x, dim=[2, 3])  # Global average pooling
        return torch.sigmoid(x).view(-1, 1)
