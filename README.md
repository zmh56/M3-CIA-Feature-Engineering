# Multi-Modal Cognitive Impairment Assessment (M3-CIA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange.svg)](https://github.com/your-username/M3-CIA)

A comprehensive deep learning framework for cognitive impairment assessment using wearable physiological and behavioral signals. This repository implements a cascaded neural network architecture with DTA (Dynamic Task-Adaptive) encoders for multi-modal fusion, enabling early detection and classification of cognitive impairment including Mild Cognitive Impairment (MCI) and Dementia (DE).

## 🚀 Key Features

- **Multi-Modal Signal Processing**: Support for EEG, ECG, Speech, Video, and Task-based behavioral data
- **DTA Encoder Architecture**: Task-Specific Attention mechanisms for time-series data processing
- **Comprehensive Feature Engineering**: Multi-dimensional feature space across different modalities
- **Cascaded Classification**: Multi-level supervision with intermediate and final outputs
- **Clinical-Ready Framework**: Designed for cognitive assessment in clinical settings

## 📊 Supported Modalities

| Modality | Description | Features | Implementation |
|----------|-------------|----------|----------------|
| **EEG** | Electroencephalography signals from cognitive tasks | N_EEG features (per-task features × multiple tasks) | Statistical, spectral, non-linear analysis |
| **ECG** | Electrocardiography signals with HRV analysis | N_ECG features | Time/frequency domain, non-linear, entropy measures |
| **Speech** | Audio features from Cookie Theft picture description | N_Speech features | OpenSMILE, WeNet ASR, BERT-Chinese |
| **Video** | Facial expression analysis using Action Units | N_Video features (AU features × multiple segments) | OpenFace, FACS-based analysis |
| **Task** | Task-based cognitive assessment scores | N_Task features | Neuropsychological measures |
| **Base** | Demographic and baseline information | N_Base features | Age, gender, education, etc. |

## 🏗️ Architecture

### DTA-Based Multi-Modal Fusion Framework
```
Input: Multi-modal signals (N_total features)
    ↓
Modality-Specific Processing:
    ├── EEG/ECG/Video → Linear Mapping → DTA Encoder → D_embed
    ├── Task/Speech/Base → FC Layers → D_embed
    ↓
Weighted Fusion (Learnable Weights)
    ↓
Cascaded Classification:
    ├── Intermediate Supervision (N_intermediate classes)
    └── Final Classification (N_classes: NC / MCI&Dementia)
```

### Key Components

- **DTA Encoder**: Dynamic Task-Adaptive encoder for time-series data
- **Modality Fusion**: Learnable weighted combination of modality representations
- **Cascaded Supervision**: Multi-level training with intermediate and final outputs
- **Task-Specific Attention**: Adaptive attention mechanisms for different cognitive tasks

### Symbol Variables

| Variable | Description | Typical Values |
|----------|-------------|----------------|
| `N_total` | Total feature dimension across all modalities | Configurable |
| `N_EEG` | EEG feature dimension | Per-task features × number of tasks |
| `N_ECG` | ECG feature dimension | HRV and morphological features |
| `N_Speech` | Speech feature dimension | Acoustic + paralinguistic + semantic features |
| `N_Video` | Video feature dimension | Action Units × number of segments |
| `N_Task` | Task feature dimension | Neuropsychological measures |
| `N_Base` | Base feature dimension | Demographic and baseline features |
| `D_embed` | Embedding dimension for each modality | Configurable (e.g., 8) |
| `N_intermediate` | Number of intermediate supervision classes | Configurable (e.g., 7) |
| `N_classes` | Number of final classification classes | 2 (NC/MCI/Dementia) |

## 📁 Project Structure

```
M3-CIA/
├── 📁 multimodal_cia/                    # Main Python package
│   ├── 📁 models/                        # Deep learning models
│   │   ├── multimodal_model.py           # Multi-modal fusion model
│   │   └── dtase_encoder.py              # DTA encoder implementation
│   ├── 📁 data/                          # Data processing utilities
│   │   └── mat_loader.py                 # MATLAB data loader
│   ├── 📁 training/                      # Training framework
│   │   └── trainer.py                    # Training pipeline
│   └── 📁 evaluation/                    # Evaluation metrics
│       └── metrics.py                    # Comprehensive evaluation metrics
├── 📁 scripts/                           # Main execution scripts
│   ├── train.py                          # Training script
│   ├── evaluate.py                       # Model evaluation script
│   └── generate_obfuscated_data.py       # Data generation script
├── 📁 examples/                          # Usage examples
│   ├── basic_usage.py                    # Basic usage example
│   └── README.md                         # Examples documentation
├── 📁 tests/                             # Unit tests
│   ├── test_models.py                    # Model tests
│   ├── test_data_loader.py               # Data loader tests
│   └── README.md                         # Testing documentation
├── 📁 configs/                           # Configuration files
│   ├── default_config.json               # Default configuration
│   └── README.md                         # Configuration documentation
├── 📁 Feature Engineering/               # Feature documentation
│   ├── eeg_features_detailed.md          # EEG feature descriptions
│   ├── ecg_features_detailed.md          # ECG feature descriptions
│   ├── speech_features_detailed.md       # Speech feature descriptions
│   └── facial_expression_features_detailed.md # Video feature descriptions
├── 📁 data_pre_mat/                      # Training data
│   └── data.mat                          # Multi-modal dataset
├── 📁 checkpoints/                       # Model checkpoints
│   └── experiment_*/                     # Experiment-specific checkpoints
├── requirements.txt                      # Python dependencies
├── setup.py                              # Package installation script
├── config.py                             # Configuration management
├── LICENSE                                # MIT License
├── .gitignore                            # Git ignore rules
└── README.md                             # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- 8GB+ RAM recommended
- MATLAB (for data preprocessing, optional)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/M3-CIA.git
cd M3-CIA

# Create virtual environment
conda create -n m3cia python=3.8
conda activate m3cia

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install additional dependencies
pip install numpy pandas scikit-learn matplotlib seaborn
pip install scipy imbalanced-learn shap
pip install h5py  # for MATLAB .mat file support

# Install package in development mode
pip install -e .
```

### Dependencies
- **Core ML**: PyTorch, NumPy, Pandas, Scikit-learn
- **Data Processing**: SciPy, H5py, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Model Interpretation**: SHAP
- **Evaluation**: Custom metrics implementation

## 🚀 Quick Start

### 1. Run Basic Example
```bash
# Run the basic usage example
cd examples
python basic_usage.py
```

### 2. Train the Model
```bash
# Simple training with default parameters
python scripts/train.py --data_path ./data/data.mat --num_epochs 20 --batch_size 16

# Advanced training with custom parameters
python scripts/train.py \
    --data_path ./data/data.mat \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --intermediate_dim 7 \
    --save_dir ./checkpoints
```

### 2. Evaluate the Model
```bash
# Evaluate trained model
python scripts/evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path ./data/data.mat \
    --output_dir ./results
```

### 3. Configuration Management
```bash
# Use custom configuration
python scripts/train.py --config configs/my_config.json

# Create default configuration
python config.py
```

### 4. Programmatic Usage
```python
from multimodal_cia.models import MultiModalModel
from multimodal_cia.data import create_mat_data_loaders
from multimodal_cia.training import Trainer

# Initialize the DTA-based multi-modal model
model = MultiModalModel(
    input_dim=N_total,      # Total feature dimension
    num_classes=N_classes,       # NC, MCI, Dementia
    intermediate_dim=N_intermediate   # Intermediate supervision classes
)

# Load data
train_loader, val_loader, test_loader = create_mat_data_loaders(
    data_path='./data/data.mat',
    batch_size=16,
    test_split=0.2,
    val_split=0.2
)

# Configure training
trainer = Trainer(
    model=model,
    learning_rate=0.001,
    batch_size=16,
    num_epochs=100,
    early_stopping=True,
    save_dir='./checkpoints'
)

# Start training
trainer.train(train_loader, val_loader)
```

## 🧪 Testing

### Run Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_models.py
python -m pytest tests/test_data_loader.py

# Run with coverage
python -m pytest tests/ --cov=multimodal_cia --cov-report=html
```

### Test Structure
- **Model Tests**: Test model creation, forward pass, parameter counting
- **Data Loader Tests**: Test data loading, batch creation, error handling
- **Integration Tests**: Test complete workflows

## 📈 Model Architecture Details

### DTA Encoder Implementation
The DTA (Dynamic Task-Adaptive) encoder is a key innovation in this framework:

```python
class TaskSEEncoder(nn.Module):
    def __init__(self, feat_dim: int, num_tasks: int, reduction: int = 4):
        """
        DTA encoder for time-series data processing
        - Input: [B, T, D] where T = num_tasks, D = feat_dim
        - Squeeze: Global average pooling across feature dimension
        - Excitation: Two-layer MLP with ReLU and Sigmoid
        - Rescale: Element-wise multiplication with gating weights
        - Output: Task-adaptive feature representations
        """
```

### Feature Processing Pipeline
1. **Input Processing**: N_total-dimensional feature vector
2. **Modality Separation**: EEG (N_EEG), ECG (N_ECG), Speech (N_Speech), Video (N_Video), Task (N_Task), Base (N_Base)
3. **DTA Encoding**: Time-series modalities (EEG/ECG/Video) → D_embed representations
4. **FC Processing**: Non-temporal modalities (Task/Speech/Base) → D_embed representations
5. **Weighted Fusion**: Learnable modality weights → (6 × D_embed)-dimensional combined representation
6. **Cascaded Classification**: Intermediate (N_intermediate classes) → Final (N_classes)

### Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Functions**: BCEWithLogitsLoss (intermediate), CrossEntropyLoss (final)
- **Early Stopping**: Patience of 10 epochs
- **Batch Size**: 16 (adjustable)
- **Epochs**: 100 (with early stopping)

### Configuration Flexibility
The framework supports flexible configuration of feature dimensions and model architecture:

```python
# Example configuration
config = {
    'feature_dims': {
        'eeg': N_EEG,      # EEG feature dimension
        'ecg': N_ECG,      # ECG feature dimension  
        'speech': N_Speech, # Speech feature dimension
        'video': N_Video,   # Video feature dimension
        'task': N_Task,     # Task feature dimension
        'base': N_Base      # Base feature dimension
    },
    'model_params': {
        'embed_dim': D_embed,           # Embedding dimension
        'intermediate_dim': N_intermediate,  # Intermediate classes
        'num_classes': N_classes,       # Final classification classes
    }
}
```

## 🔬 Research Applications

This framework is designed for comprehensive cognitive assessment research:

### Clinical Applications
- **Early Detection**: Identify cognitive impairment before clinical symptoms
- **Differential Diagnosis**: Distinguish between normal and impaired cognition
- **Progression Monitoring**: Track cognitive changes over time
- **Treatment Evaluation**: Assess intervention effectiveness

### Research Domains
- **Digital Biomarkers**: Non-invasive assessment using wearable devices
- **Multi-Modal Analysis**: Integrate physiological and behavioral signals
- **Longitudinal Studies**: Long-term cognitive health monitoring
- **Population Studies**: Large-scale cognitive screening programs

### Technical Advantages
- **Reproducible Pipeline**: Complete experimental setup with synthetic data
- **Modular Architecture**: Easy to extend with new modalities
- **Clinical Validation**: Designed for real-world clinical applications
- **Open Source**: Transparent and extensible implementation

## 📚 Documentation

### Feature Engineering Documentation
- [EEG Features](Feature%20Engineering/eeg_features_detailed.md) - Per-task features, statistical/spectral/non-linear analysis
- [ECG Features](Feature%20Engineering/ecg_features_detailed.md) - HRV features, time/frequency/entropy measures
- [Speech Features](Feature%20Engineering/speech_features_detailed.md) - Acoustic/paralinguistic/semantic features
- [Facial Expression Features](Feature%20Engineering/facial_expression_features_detailed.md) - Action Units with OpenFace

### Technical Documentation
- **Model Architecture**: DTA encoder with cascaded classification
- **Data Format**: MATLAB .mat files with standardized feature extraction
- **Training Pipeline**: End-to-end training with early stopping and validation
- **Evaluation Metrics**: Comprehensive classification and regression metrics

### Implementation Details
- **OpenSMILE**: Acoustic feature extraction (eGeMAPSV02)
- **WeNet**: End-to-end speech recognition
- **BERT-Chinese**: Semantic analysis and similarity computation
- **OpenFace**: Facial Action Unit detection and analysis

## 🤝 Contributing

We welcome contributions from the research community! 

### Development Setup
```bash
# Install development dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn
pip install scipy imbalanced-learn shap h5py

# Clone and setup
git clone https://github.com/your-username/M3-CIA.git
cd M3-CIA
pip install -e .

# Run basic tests
python scripts/train.py --num_epochs 1 --batch_size 16
```

### Contribution Areas
- **New Modalities**: Extend framework with additional signal types
- **Feature Engineering**: Implement new feature extraction methods
- **Model Architecture**: Improve DTA encoder or fusion mechanisms
- **Evaluation**: Add new metrics or validation strategies
- **Documentation**: Improve feature descriptions and tutorials

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{m3cia2024,
  title={Multi-Modal Deep Learning Framework for Cognitive Impairment Assessment Using Wearable Physiological and Behavioral Signals},
  author={[Your Name] and [Co-authors]},
  journal={[Target Journal]},
  year={2024},
  publisher={[Publisher]},
  doi={[DOI when available]}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Research Support
- Research collaborators and clinical partners
- Study participants who contributed to the dataset
- Institutional support for cognitive health research

### Technical Acknowledgments
- **OpenSMILE**: Acoustic feature extraction framework
- **WeNet**: End-to-end speech recognition toolkit
- **BERT-Chinese**: Pre-trained language models for semantic analysis
- **OpenFace**: Facial behavior analysis toolkit
- **PyTorch**: Deep learning framework

### Open Source Community
- Contributors to the open-source tools and libraries used in this project
- Research community for sharing knowledge and methodologies

## 📞 Contact & Support

### Research Team
- **Lead Researcher**: [Your Name](mailto:your.email@institution.edu)
- **Institution**: [Your Institution]
- **Research Group**: [Your Research Group]

### Technical Support
- **Issues**: [GitHub Issues](https://github.com/your-username/M3-CIA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/M3-CIA/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-username/M3-CIA/wiki)

## ⚠️ Important Notes

### Research Code Disclaimer
This is research code intended for academic and research purposes. While we strive for quality and reproducibility, this software is provided "as is" without warranty of any kind.

### Clinical Applications
- **Not for Clinical Use**: This framework is designed for research purposes only
- **Clinical Validation Required**: Any clinical applications require proper validation studies
- **Healthcare Professional Consultation**: Always consult with qualified healthcare professionals for clinical decisions

### Data Privacy
- **Data Protection**: Ensure compliance with local privacy regulations when using patient data
- **IRB Approval**: Obtain appropriate institutional review board approval for human subjects research
- **Data Security**: Implement appropriate data security measures for sensitive information

---

*This work represents ongoing research in digital biomarkers for cognitive health assessment. We encourage the research community to build upon this work and contribute to advancing cognitive health technologies.*
