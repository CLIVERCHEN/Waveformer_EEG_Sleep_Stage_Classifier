import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder_1d1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=4, padding=0),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, stride=7, padding=0),
            nn.ReLU(True)
        )

        self.encoder_2d = nn.Sequential(
            # Additional layer to transform from 1D to 2D
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(True)
        )

        self.encoder_1d2 = nn.Sequential(
            # Reshape layer might be required here
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=4, padding=0),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, stride=1, padding=0),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder1d1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=6, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=9, stride=4, padding=0, output_padding=3),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.ReLU(True)
        )

        self.decoder2d = nn.Sequential(
            # Additional layer to transform from 2D to 1D
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=2, padding=1, output_padding=(1,0)),
            nn.ReLU(True)
        )

        self.decoder1d2 = nn.Sequential(
            # Reshape layer might be required here
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=15, stride=7, padding=0, output_padding=4),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=9, stride=4, padding=0, output_padding=3),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder_1d1(x)

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2, 3)
        x = self.encoder_2d(x)

        x = x.permute(1, 0, 2, 3)
        x = x.squeeze(0)
        x = self.encoder_1d2(x)

        x = self.decoder1d1(x)

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2, 3)
        x = self.decoder2d(x)

        x = x.permute(1, 0, 2, 3)
        x = x.squeeze(0)
        x = self.decoder1d2(x)

        x = F.pad(x, (0, 4))
        return x

if __name__ == "__main__":
    autoencoder = ConvAutoencoder()
    model = ConvAutoencoder()
    # 假设输入
    input_tensor = torch.randn(1, 64, 30 * 256)
    output = model(input_tensor)

