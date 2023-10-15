import torch
from torchvision import transforms
import cv2
from torch.utils.data import DataLoader
import time
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from odak.learn.wave import generate_complex_field, calculate_phase

from algorithms.SGD import SGD
from utils.dataset import Div2k


img_dir = 'data/test'
transform = transforms.Compose([
    transforms.Resize(size=(1080, 1920)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])
test_set = Div2k(img_dir, transform)
img = next(iter(test_set))
img = img.unsqueeze(-1)
img = img.unsqueeze(0)
print(f'target image size: {img.shape}')

# target shape for SGD: B×C×H×W×D
model = SGD(wavelengths=[520e-9],
            # wavelengths=[638e-9, 520e-9, 450e-9],
            pixel_size=6.4e-6,
            resolution=[1080, 1920],
            distances=[20e-2],
            target=img,
            learning_rate=2e-1,
            number_of_frame=1)

# SGD iteration
data = torch.rand(10)
logger = TensorBoardLogger(save_dir='lightning_logs', name='test')
trainer = Trainer(max_epochs=20, accelerator='cuda', log_every_n_steps=1, logger=logger)

# post-process save hologram
trainer.fit(model, train_dataloaders=data)
init_phase = model.phase.detach()   # range -10 ~ 10
phase_only_hologram = torch.atan2(torch.sin(init_phase), torch.cos(init_phase))
phase_only_hologram = phase_only_hologram.squeeze(0).permute(1, 2, 0).numpy()
phase_only_hologram += torch.pi     # range 0 ~ 2pi
print(phase_only_hologram.max(), phase_only_hologram.min())
cv2.imwrite('phase_only_hologram.png', phase_only_hologram / phase_only_hologram.max() * 255.0)
print(f'Phase only hologram shape: {phase_only_hologram.shape}')
