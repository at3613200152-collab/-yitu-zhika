import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.block(x)
        x = torch.cat([x, skip_input], dim=1)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, num_downs=7):
        super().__init__()
        
        # Encoder: 7 downsampling blocks
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2
        self.down1 = UNetDownBlock(in_channels, base_channels, use_bn=False)       # 256->128, ch=64
        self.down2 = UNetDownBlock(base_channels, base_channels * 2)               # 128->64,  ch=128
        self.down3 = UNetDownBlock(base_channels * 2, base_channels * 4)           # 64->32,   ch=256
        self.down4 = UNetDownBlock(base_channels * 4, base_channels * 8)           # 32->16,   ch=512
        self.down5 = UNetDownBlock(base_channels * 8, base_channels * 8)           # 16->8,    ch=512
        self.down6 = UNetDownBlock(base_channels * 8, base_channels * 8)           # 8->4,     ch=512
        self.down7 = UNetDownBlock(base_channels * 8, base_channels * 8)           # 4->2,     ch=512

        # Bottleneck: 2->1 (stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder: 8 upsampling steps to match 8 downsampling steps
        # 1->2->4->8->16->32->64->128->256
        self.up7 = UNetUpBlock(base_channels * 8, base_channels * 8, use_dropout=True)    # 1->2, out=512, concat d7(512) => 1024
        self.up6 = UNetUpBlock(base_channels * 16, base_channels * 8, use_dropout=True)   # 2->4, out=512, concat d6(512) => 1024
        self.up5 = UNetUpBlock(base_channels * 16, base_channels * 8, use_dropout=True)   # 4->8, out=512, concat d5(512) => 1024
        self.up4 = UNetUpBlock(base_channels * 16, base_channels * 4)                     # 8->16, out=256, concat d4(512) => 768
        self.up3 = UNetUpBlock(base_channels * 12, base_channels * 2)                    # 16->32, out=128, concat d3(256) => 384
        self.up2 = UNetUpBlock(base_channels * 6, base_channels)                          # 32->64, out=64, concat d2(128) => 192
        self.up1 = UNetUpBlock(base_channels * 3, base_channels)                          # 64->128, out=64, concat d1(64) => 128
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),  # 128->256
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bottleneck = self.bottleneck(d7)

        u7 = self.up7(bottleneck, d7)
        u6 = self.up6(u7, d6)
        u5 = self.up5(u6, d5)
        u4 = self.up4(u5, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        out = self.up0(u1)

        return out


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, num_layers=3):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        curr_channels = base_channels
        for i in range(num_layers):
            next_channels = min(base_channels * (2 ** (i + 1)), 512)
            stride = 1 if i == num_layers - 1 else 2
            layers.extend([
                nn.Conv2d(curr_channels, next_channels, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            curr_channels = next_channels
        
        layers.append(nn.Conv2d(curr_channels, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Pix2PixModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["models"]["nir_generator"]
        
        self.generator = UNetGenerator(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            base_channels=model_cfg["base_channels"],
            num_downs=model_cfg["num_downs"],
        )
        
        self.discriminator = PatchGANDiscriminator(
            in_channels=model_cfg["in_channels"] + model_cfg["out_channels"],
            base_channels=model_cfg["base_channels"],
        )
        
        self.lambda_l1 = model_cfg["lambda_l1"]
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, real_rgb):
        fake_nir = self.generator(real_rgb)
        return fake_nir
    
    def compute_discriminator_loss(self, real_rgb, real_nir, fake_nir):
        fake_input = torch.cat([real_rgb, fake_nir.detach()], dim=1)
        real_input = torch.cat([real_rgb, real_nir], dim=1)
        
        pred_fake = self.discriminator(fake_input)
        pred_real = self.discriminator(real_input)
        
        loss_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_real = self.gan_loss(pred_real, torch.ones_like(pred_real))
        loss_d = (loss_fake + loss_real) * 0.5
        
        return loss_d
    
    def compute_generator_loss(self, real_rgb, real_nir, fake_nir):
        fake_input = torch.cat([real_rgb, fake_nir], dim=1)
        pred_fake = self.discriminator(fake_input)
        
        loss_gan = self.gan_loss(pred_fake, torch.ones_like(pred_fake))
        loss_l1 = self.l1_loss(fake_nir, real_nir)
        
        loss_g = loss_gan + self.lambda_l1 * loss_l1
        
        return loss_g, loss_gan, loss_l1
