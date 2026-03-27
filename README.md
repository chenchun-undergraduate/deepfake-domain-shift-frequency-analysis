# Robust Deepfake Detection in Compressed Environments: Domain Shift and Frequency Analysis
This project investigates how video compression affects deepfake detection from a frequency-domain perspective. 
We analyze FFT and DCT spectra under different compression levels and show that the degradation in model performance is primarily caused by domain shift rather than a simple loss of discriminative information.
## Research Questions

1. Do frequency-domain changes introduced by video compression constitute domain shift?
2. Can reducing model reliance on unstable frequency features improve generalization across compression levels?

## Dataset

We use the FaceForensics++ dataset, including:
- original
- CRF28
- CRF35
- mixed compression setting

To avoid data leakage, different compression versions of the same video are assigned to the same train/test split.

## Methodology

### 1. Frequency Analysis
- FFT spectrum comparison
- DCT spectrum comparison
- radial frequency energy
- compression difference maps

### 2. Model Training
- Backbone: ResNet18
- Pretrained on ImageNet
- Full fine-tuning
- Mixed-compression training
- Baseline: original → original

## Main Findings

- Compression does not simply remove high-frequency information.
- Instead, it changes the distribution of frequency-domain features.
- Models trained on a single compression level suffer from cross-domain degradation.
- Mixed-compression training significantly improves robustness.
- This suggests that the main issue is domain shift rather than intrinsic feature loss.
