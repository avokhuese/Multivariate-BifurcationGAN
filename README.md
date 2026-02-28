#BifurcationGAN: Advanced Time Series Augmentation Framework for Multivariate and Univariate Data

## 📋 Title
**BifurcationGAN: A generative adversarial network framework for multivariate time series augmentation informed by hopf bifurcation dynamics**

## 📝 Description

This repository contains a comprehensive framework for time series data augmentation using novel Generative Adversarial Network architectures. The core innovation is the **BifurcationGAN**, which combines dynamical systems theory with deep learning to generate high-quality synthetic time series data. The study implements 4 state-of-the-art GAN variants for comprehensive benchmarking with the novel BifurcationGAN model with bifurcation dynamics:

**Novel Architectures:**
1. **BifurcationGAN** - GAN with Hopf bifurcation dynamics

**Benchmark Models:**
1. Vanilla GAN
2. WGAN (Wasserstein GAN)
3. WGAN-GP (WGAN with Gradient Penalty)
4. TTS-GAN (Time Series Synthesis GAN)



The framework supports both **multivariate** and **univariate** time series and is evaluated on 15+ diverse benchmark datasets from the aeon library.

## 📊 Dataset Information

### Multivariate Datasets (15)
The framework supports 15 multivariate time series datasets from the aeon library:

| Dataset | Samples | Features | Length | Domain |
|---------|---------|----------|--------|--------|
| BasicMotions | 80 | 6 | 100 | Human Activity |
| EigenWorms | 259 | 6 | 17984 | Biology |
| Epilepsy | 275 | 3 | 206 | Healthcare |
| ERing | 300 | 4 | 65 | Gesture Recognition |
| FaceDetection | 5890 | 144 | 62 | Computer Vision |
| FingerMovements | 416 | 28 | 50 | Neuroscience |
| HandMovementDirection | 234 | 10 | 400 | Gesture Recognition |
| Handwriting | 150 | 3 | 152 | Handwriting Recognition |
| Heartbeat | 409 | 61 | 405 | Healthcare |
| JapaneseVowels | 640 | 12 | 29 | Speech Recognition |
| Libras | 360 | 2 | 45 | Sign Language |
| LSST | 4925 | 6 | 36 | Astronomy |
| MotorImagery | 378 | 64 | 3000 | Neuroscience |
| NATOPS | 360 | 24 | 51 | Gesture Recognition |
| PEMS-SF | 440 | 963 | 144 | Traffic |


## 💻 Code Information

### Directory Structure

# O-BGAN/
- ├── config_multivariate.py # Configuration for multivariate experiments
- ├── config_univariate.py # Configuration for univariate experiments
- ├── data_loader_multivariate.py # Data loading for multivariate datasets
- ├── data_loader_multivariate_fixed.py # Fixed-size data loader for multivariate
- ├── data_loader_univariate.py # Data loading for univariate datasets
- ├── data_loader_univariate_fixed.py # Fixed-size data loader for univariate
- ├── models_multivariate.py # Multivariate GAN architectures
- ├── models_univariate.py # Univariate GAN architectures
- ├── baseline_models_multivariate.py # Baseline GAN implementations for multivariate
- ├── baseline_models_univariate.py # Baseline GAN implementations for univariate
- ├── gan_framework_multivariate.py # Training framework for multivariate
- ├── gan_framework_univariate.py # Training framework for univariate
- ├── evaluation_multivariate.py # Evaluation metrics for multivariate
- ├── evaluation_univariate.py # Evaluation metrics for univariate
- ├── training_dynamics.py # Training dynamics visualization
- ├── ablation_study.py # Component ablation analysis
- ├── visualizations.py # Publication-quality figure generation
- ├── debug_dataset.py # Dataset debugging utilities
- ├── main_multivariate_pipeline.py # Main pipeline for multivariate experiments
- ├── main_univariate_pipeline.py # Main pipeline for univariate experiments
- ├── run_multivariate.py # Entry point for multivariate experiments
- ├── run_univariate.py # Entry point for univariate experiments
- ├── run_ablation.py # Ablation study runner
- ├── run_training_analysis.py # Training dynamics analysis
- ├── generate_paper_figures.py # Paper figure generation
- ├── requirements.txt # Dependencies
- └── README.md # This file


### Key Components

#### Configuration (`config_*.py`)
Central configuration management with dataclasses. Controls:
- Model architectures (hidden dimensions, layers)
- Bifurcation parameters (Hopf mu, omega, alpha, beta)
- Training parameters (learning rates, batch sizes, epochs)
- Evaluation metrics (FID, MMD, Wasserstein, ACF, PSD)
- Dataset-specific parameters

#### Data Loaders (`data_loader_*.py`)
- **Adaptive sequence handling**: Pads or truncates sequences to fixed length
- **Bifurcation-aware sampling**: Prioritizes dynamic regions of time series
- **Multi-scale processing**: Handles datasets with varying lengths
- **Caching system**: Saves preprocessed data for faster reloading
- **Fallback generation**: Creates synthetic data when real data unavailable

#### Model Architectures (`models_*.py`, `baseline_models_*.py`)

**BifurcationGAN Components:**
- `BifurcationDynamicsLayer`: Implements Hopf, pitchfork, saddle-node, and transcritical bifurcations
- `HierarchicalNoiseGenerator`: Multi-scale noise generation

**Novel Architectures:**
- `BifurcationGenerator`: GAN with bifurcation dynamics in latent space
- `BifurcationDiscriminator`: Multi-scale discriminator with bifurcation detection

**Baseline Models:**
- Vanilla GAN (MLP-based)
- WGAN (Weight clipping)
- WGAN-GP (Gradient penalty)
- TTS-GAN (LSTM-based)


#### Training Framework (`gan_framework_*.py`)
- **Mixed precision training**: AMP support for faster training
- **Gradient penalty**: WGAN-GP implementation
- **History tracking**: Losses, gradient norms, Wasserstein distance
- **Checkpointing**: Automatic model saving and loading
- **Multi-seed support**: Statistical significance across runs
- **Early stopping**: Prevents overfitting

#### Evaluation (`evaluation_*.py`)
**Distribution Metrics:**
- Jensen-Shannon Divergence
- Kolmogorov-Smirnov Statistic
- Wasserstein Distance
- Maximum Mean Discrepancy (MMD)

**Temporal Metrics:**
- Autocorrelation Function (ACF) Similarity
- Power Spectral Density (PSD) Similarity
- Cross-correlation Similarity

**Quality Metrics:**
- Fréchet Inception Distance (FID) for time series
- Precision-Recall-Density (PRD)
- Composite Score (weighted combination)

**Bifurcation-Specific Metrics:**
- Lyapunov Exponent Similarity
- Phase Space Reconstruction Similarity
- Poincaré Map Similarity

#### Training Dynamics (`training_dynamics.py`)
- Loss curves with confidence intervals
- Gradient norm evolution
- Failure rate analysis
- Convergence speed comparison
- Stability heatmaps
- Comprehensive 4-panel dynamics figures

#### Ablation Study (`ablation_study.py`)
- Component contribution analysis
- Statistical significance testing
- p-value heatmaps
- Radar charts for multi-metric comparison


#### Visualizations (`visualizations.py`)
Publication-quality figure generation:
- Time series comparison plots
- Distribution analysis (histograms + KDE)
- Temporal dynamics (ACF, PSD, phase space)
- Model comparison bar charts
- Radar charts for multi-metric comparison
- t-SNE manifold visualization
- Bifurcation dynamics visualization

## 🚀 Usage Instructions

### Installation

1. **Clone the repository**
 
git clone https://github.com/avokhuese/Multivariate-BifurcationGAN.git
cd Multivariate-BifurcationGAN


2. **Create virtual environment**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
   pip install -r requirements.txt

## Quick Start and Full Benchmark
python run_multivariate.py --debug
python run_multivariate.py --test
python run_multivariate.py

## Ablation Study
python run_ablation.py


## Key Innovations

1. Bifurcation-Aware Generation: Models learn to generate data that follows dynamical system principles
2. Coupled Oscillator Dynamics: Captures complex periodic and quasi-periodic behaviors
3. Multi-Scale Processing: Handles datasets with varying lengths and dynamics
4. Comprehensive Evaluation: 10+ metrics for thorough assessment
5. Statistical Rigor: Multiple runs with significance testing


## Requirements
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Progress and utilities
tqdm>=4.62.0

# Dataset loading
aeon>=0.6.0

# Optional
plotly>=5.3.0  # Interactive visualizations
psutil>=5.8.0  # System monitoring
jupyter>=1.0.0 # Notebook support


## For the datasets, please cite the aeon library
@misc{middlehurst2024aeonpythontoolkitlearning,
      title={aeon: a Python toolkit for learning from time series}, 
      author={Matthew Middlehurst and Ali Ismail-Fawaz and Antoine Guillaume and Christopher Holder and David Guijo Rubio and Guzal Bulatova and Leonidas Tsaprounis and Lukasz Mentel and Martin Walter and Patrick Schäfer and Anthony Bagnall},
      year={2024},
      eprint={2406.14231},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.14231}, 
}


# Contact
For questions or collaborations

- Email: avokhuese@gmail.com or alexander.victor4@mail.dcu.ie
- Github: @avokhuese
