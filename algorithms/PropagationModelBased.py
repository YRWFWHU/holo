import torch
from odak.learn.wave import get_band_limited_angular_spectrum_kernel, zero_pad, crop_center, custom


class PropagationModel:
    def __init__(self, wavelengths, pixel_size, resolution, distances):
        """

        :param wavelengths: list, ordered in RGB
        :param pixel_size: float,
        :param resolution: list, H,W
        :param distances: list, multi-plane distance
        """
        self.wavelengths = wavelengths
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.distances = distances

        self.asm_kernel = torch.randn(
            len(self.wavelengths),
            len(self.distances),
            self.resolution[-2] * 2,  # *2 for zero padding
            self.resolution[-1] * 2,
            dtype=torch.complex64,
        )
        for idx_wave, wave in enumerate(self.wavelengths):
            for idx_dist, dist in enumerate(self.distances):
                self.asm_kernel[idx_wave, idx_dist] = get_band_limited_angular_spectrum_kernel(
                    resolution[-2] * 2, resolution[-1] * 2, dx=pixel_size, wavelength=wave, distance=dist)

    def reconstruct(self, hologram):
        """
        reconstruct hologram
        :param hologram: input hologram, shape is B×T×H×W
        :return: reconstructed img, shape is B×C×D×H×W
        """
        hologram = zero_pad(hologram)
        recon_intensity = torch.ones(
            hologram.shape[0],
            len(self.wavelengths),
            self.resolution[-2],
            self.resolution[-1],
            len(self.distances),
        )
        for idx_wave, wave in enumerate(self.wavelengths):
            for idx_dist, dist in enumerate(self.distances):
                # multi-frame complex: B×T×H×W
                recon_complex = crop_center(custom(hologram, self.asm_kernel[idx_wave, idx_dist], zero_padding=False,
                                                   aperture=1.))
                recon_intensity[:, idx_wave, :, :, idx_dist] = torch.sum(torch.abs(recon_complex) ** 2, dim=1)
        return recon_intensity
