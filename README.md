# KWS_Text
Audio-Text Cross-Modality Learning for Small-footprint Keyword Spotting.

# Install
easydict==1.9

numpy==1.19.1

torch==1.6.0+cu101

torchsummaryx==1.3.0

torchaudio==0.6.0

tqdm==4.48.2

# Usage
Change config.py to train different models in different modes.
## Dataset
Support Google Speech Commands Dataset v1 and v2.
## Set up
MODELTYPE: The type of model. (e.g. LGNet6)

LOSS: Loss function, ['CE'] for LGNet6, ['CE', 'TRIPLET'] for LGNet6_ThreeAudios.

MODE: Train or test mode.

# Maintainers
@WWWWWLI
