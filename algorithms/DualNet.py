import itertools
import torch
import torchmetrics.image
from torch import nn
import torch.fft
from complexPyTorch.complexLayers import ComplexConvTranspose2d, ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
from utils.propagation_ASM import propagation_ASM
import pytorch_lightning as pl
import torchvision


class CDown(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super().__init__()
        self.Conv = ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.relu = relu

    def forward(self, x):
        if self.relu:
            out = complex_relu((self.Conv(x)))
        else:
            out = self.Conv(x)
        return out


class CUp(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super().__init__()
        self.Conv = ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.relu = relu

    def forward(self, x):
        if self.relu:
            out = complex_relu((self.Conv(x)))
        else:
            out = self.Conv(x)
        return out


class PhaseGenerator(nn.Module):
    """
    input shape: B×C×H×W complex tensor
    output shape: B×C×H×W predict phase [-pi，pi]
    """

    def __init__(self, input_channel, output_channel, stage=3):
        super().__init__()
        self.down_sampler = nn.ModuleList()
        self.up_sampler = nn.ModuleList()
        self.input_projection = CDown(input_channel, 4)
        for i in range(stage):
            self.down_sampler.append(CDown(4 * (2 ** i), 8 * (2 ** i)))
            self.up_sampler.insert(0, CUp(8 * (2 ** i), 4 * (2 ** i)))

        self.up_sampler.append(CUp(4, output_channel, False))

    def forward(self, x):
        skip_connection = []
        x = self.input_projection(x)
        skip_connection.append(x)
        for layer in self.down_sampler:
            x = layer(x)
            skip_connection.append(x)

        skip_connection.pop()
        for idx, layer in enumerate(self.up_sampler):
            if idx != 0:
                x += skip_connection.pop()
            x = layer(x)

        predict_phase = torch.atan2(x.imag, x.real)
        return predict_phase


class CCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)

        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp(4, 1, False)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)

        out18 = self.netup3(out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)

        holophase = torch.atan2(out20.imag, out20.real)
        return holophase


class DualNet(pl.LightningModule):
    """
    input_channel: 3 for RGB, 1 for Grayscale
    output_channel: 1 for single hologram, 2 for Dual hologram
    wavelength: list [] 3 element for Full Color, 1 element for Grayscale
    lr: learning rate, default 1e-3
    """

    def __init__(self, z=1.5e2, pad=False, pixel_size=3.6e-3, wavelength=(6.38e-4, 5.20e-4, 4.50e-4),
                 resolution=(1080, 1920),
                 input_channel=3, output_channel=2, lr=1e-3):
        super().__init__()
        self.target_phase_generator = PhaseGenerator(stage=3, input_channel=input_channel, output_channel=input_channel)
        self.slm_phase_generator = PhaseGenerator(stage=2, input_channel=input_channel, output_channel=output_channel)
        self.loss = nn.MSELoss()
        self.init_phase = torch.zeros(1, len(wavelength), resolution[0], resolution[1])
        self.z = z
        self.pad = pad
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.lr = lr
        self.Hforward = []
        self.Hbackward = []
        for wave in self.wavelength:
            self.Hforward.append(propagation_ASM(torch.empty(1, 1, resolution[0], resolution[1]),
                                                 feature_size=[self.pixel_size, self.pixel_size],
                                                 wavelength=wave, z=-self.z, linear_conv=self.pad, return_H=True))
            self.Hbackward.append(propagation_ASM(torch.empty(1, 1, resolution[0], resolution[1]),
                                                  feature_size=[self.pixel_size, self.pixel_size],
                                                  wavelength=wave, z=self.z, linear_conv=self.pad, return_H=True))
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def training_step(self, batch, batch_idx):
        recon_intensity, target_amp, slm_phase, target_phase = self.forward(batch)
        loss = self.loss(recon_intensity, batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        recon_intensity, target_amp, slm_phase, target_phase = self.forward(batch)
        loss = self.loss(recon_intensity, batch)
        ssim_value = self.ssim(batch, recon_intensity / recon_intensity.max())
        psnr_value = self.psnr(batch, recon_intensity / recon_intensity.max())
        self.log('val_loss', loss, batch_size=1, sync_dist=True, prog_bar=True)
        self.log('val_ssim', ssim_value, batch_size=1, sync_dist=True, prog_bar=True)
        self.log('val_psnr', psnr_value, batch_size=1, sync_dist=True, prog_bar=True)
        if batch_idx == 0:
            input_images = batch[0]  # 取验证集中的前两张图像作为示例输入
            output_images = recon_intensity[0]  # 取对应的前两张模型输出作为示例输出
            input_grid = torchvision.utils.make_grid(input_images)
            output_grid = torchvision.utils.make_grid(output_images)
            self.logger.experiment.add_image("input_images", input_grid, self.current_epoch, dataformats="CHW")
            self.logger.experiment.add_image("reconstructed_images", output_grid, self.current_epoch, dataformats="CHW")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            itertools.chain(self.target_phase_generator.parameters(), self.slm_phase_generator.parameters()),
            lr=self.lr)
        return optimizer

    def forward(self, input_intensity):
        target_amp = torch.sqrt(input_intensity)
        slm_phase, target_phase = self.predict_phase(target_amp, self.init_phase)
        slm_complex = torch.complex(torch.cos(slm_phase), torch.sin(slm_phase))
        recon_intensity_single_wavelength = []
        for idx, wave in enumerate(self.wavelength):
            recon_complex = propagation_ASM(u_in=slm_complex, z=-self.z, linear_conv=self.pad,
                                            feature_size=[self.pixel_size, self.pixel_size],
                                            wavelength=self.wavelength[idx],
                                            precomped_H=self.Hbackward[idx])
            recon_amp = torch.abs(recon_complex)
            recon_intensity_dual = recon_amp ** 2
            recon_intensity_single_wavelength.append(torch.sum(recon_intensity_dual, dim=1, keepdim=True))
            # print(f'recon_intensity_single_wavelength[0]\'s shape:{recon_intensity_single_wavelength[0].shape}')
        recon_intensity = torch.cat(recon_intensity_single_wavelength, dim=1)
        return recon_intensity, target_amp, slm_phase, target_phase

    def predict_phase(self, amp, phase):
        target_complex = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
        target_phase = self.target_phase_generator(target_complex)
        target_complex = torch.complex(amp * torch.cos(target_phase), amp * torch.sin(target_phase))

        slm_complex_single_wavelength = []
        for idx, wave in enumerate(self.wavelength):
            slm_complex_single_wavelength.append(
                propagation_ASM(u_in=target_complex[:, idx:idx + 1, :, :], z=self.z, linear_conv=self.pad,
                                feature_size=[self.pixel_size, self.pixel_size],
                                wavelength=self.wavelength[idx],
                                precomped_H=self.Hforward[idx]))
        slm_complex = torch.cat(slm_complex_single_wavelength, dim=1)
        slm_phase = self.slm_phase_generator(slm_complex)
        return slm_phase, target_phase
