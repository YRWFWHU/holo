import torch
import torchmetrics
from odak.learn.wave import generate_complex_field
import pytorch_lightning as pl

from .PropagationModelBased import PropagationModel


class SGD(pl.LightningModule):
    def __init__(self, wavelengths, pixel_size, resolution, distances, target, learning_rate, number_of_frame):
        """
        :param wavelengths:
        :param pixel_size:
        :param resolution:
        :param distances:
        :param target: tensor, shape is B×C×H×W×D
        :param learning_rate:
        :param number_of_frame:
        """
        torch.manual_seed(42)
        super().__init__()
        self.prop_model = PropagationModel(wavelengths, pixel_size, resolution, distances)
        self.target = target
        # shape of phase: B×T×H×W
        self.phase = torch.randn(
            target.shape[0],
            number_of_frame,
            resolution[-2],
            resolution[-1],
            requires_grad=True
        )
        self.lr = learning_rate
        self.loss = torch.nn.MSELoss()
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def training_step(self, batch):
        hologram = generate_complex_field(1., self.phase)
        reconstruction = self.prop_model.reconstruct(hologram)
        self.log('Max Value of Reconstructed Image', reconstruction.max())
        reconstruction = reconstruction / 1.414
        loss = self.loss(reconstruction, self.target)
        self.log('MSE loss', loss, prog_bar=True)
        self.log('PSNR', self.psnr(reconstruction[:, :, :, :, 0], self.target[:, :, :, :, 0]), prog_bar=True)
        self.log('SSIM', self.ssim(reconstruction[:, :, :, :, 0], self.target[:, :, :, :, 0]), prog_bar=True)
        self.logger.experiment.add_image('Reconstructed Image', reconstruction[0, :, :, :, 0],
                                         global_step=self.global_step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([self.phase], lr=self.lr)
        return optimizer
