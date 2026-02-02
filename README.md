# QuPID

Quantum Parameter-efficient In-storage Data-dependent RAG for Medical Diagnostic Large Vision-Language Models

## Overview

QuPID is a hybrid quantum-classical retrieval framework that projects frozen vision transformer features into a quantum Hilbert space via parameterized quantum circuits, achieving superior pathological separability with only 30 trainable parameters.

## Requirements
```bash
pip install torch torchvision timm pennylane scikit-learn matplotlib pandas tqdm
```

## Dataset

Download ChestX-ray14 from [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) and organize as:
```
ChestXray14/
├── images_001/images/*.png
├── ...
├── images_012/images/*.png
├── Data_Entry_2017.csv
├── train_val_list.txt
└── test_list.txt
```

## Usage
```bash
python qupid_chestxray.py
```

## Configuration

Edit `CONFIG` dictionary in the script to modify:
- `training_mode`: `'two_stage'` or `'end_to_end'`
- `n_qubits`: Number of qubits (default: 10)
- `n_qlayers`: Number of quantum layers (default: 3)
- `temperature`: Contrastive loss temperature (default: 0.07)

## Citation
```bibtex
@inproceedings{qupid2025,
  title={QuPID: Quantum Parameter-Efficient In-Storage Data-Dependent RAG for Medical Diagnostic Large Vision-Language Models},
  author={...},
  booktitle={KDD},
  year={2025}
}
```

## License

MIT License
