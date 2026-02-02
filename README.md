# QuPID

**Quantum Parameter-efficient In-storage Data-dependent RAG for Medical Diagnostic Large Vision-Language Models**

*KDD 2025*

## Overview

QuPID is a hybrid quantum-classical retrieval framework for medical retrieval-augmented generation (RAG). It projects frozen Vision Transformer (ViT) features into a quantum Hilbert space via parameterized quantum circuits (PQC), achieving superior pathological separability with only **30 trainable parameters**.

### Key Features

- **Extreme Parameter Efficiency**: 30 parameters vs. 500K+ for classical adapters
- **Quantum Geometric Separation**: Leverages Fubini-Study metric in Hilbert space to separate overlapping pathological features
- **Privacy-Preserving**: Enables in-storage, data-dependent adaptation on local hospital data without external transfer
- **Classical Simulation**: No physical quantum hardware required

## Architecture
```
Input Image → Frozen ViT-L/16 → 1024-D Features → Amplitude Embedding (10 qubits) → PQC (3 layers) → Quantum Fidelity Retrieval
```

## Requirements
```bash
pip install torch torchvision timm pennylane scikit-learn tqdm numpy
```

## Usage
```python
from models import ViTEncoder, QuantumEnhancer, QuPIDModel

# Initialize models
vit_encoder = ViTEncoder(
    model_name='vit_large_patch16_224',
    embedding_dim=1024,
    freeze_backbone=True
)
quantum_enhancer = QuantumEnhancer(
    input_dim=1024,
    n_qubits=10,
    n_qlayers=3
)
model = QuPIDModel(vit_encoder, quantum_enhancer)

# Training
from train import Trainer, CONFIG
trainer = Trainer(CONFIG, device)
trainer.train_stage1_vit(vit_encoder, train_loader, val_loader)
trainer.train_stage2_quantum(quantum_enhancer, vit_encoder, train_loader, val_loader)
```

## Results

### Retrieval Performance (ChestX-ray14)

| Method | P@5 | MAP@10 | # Params |
|--------|-----|--------|----------|
| ViT Only | 0.312 | 0.278 | 0 |
| Classical Head | 0.378 | 0.341 | 1,051,136 |
| Adapter Layers | 0.401 | 0.368 | 525,568 |
| **QuPID** | **0.428** | **0.399** | **30** |

### Embedding Quality

| Metric | ViT Only | QuPID |
|--------|----------|-------|
| Intra-Cluster Distance ↓ | 0.684 | **0.576** |
| Inter-Cluster Distance ↑ | 0.312 | **0.416** |
| Silhouette Score ↑ | 0.182 | **0.273** |

## Datasets

Evaluated on four medical imaging benchmarks:
- **ChestX-ray14**: 112,120 chest X-rays, 14 pathologies
- **MURA**: 40,561 musculoskeletal radiographs, 14 categories
- **IU X-Ray**: 7,470 chest X-rays with radiology reports
- **MIMIC-CXR**: 377,110 chest X-rays with clinical reports
