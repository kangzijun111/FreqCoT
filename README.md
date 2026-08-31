# FreqCoT: Frequency-enhanced Semantic Community Modeling for Spatiotemporal Forecasting

This repository contains the PyTorch implementation of **FreqCoT**, a frequency-enhanced semantic community transformer for traffic forecasting.

FreqCoT organizes traffic nodes into **semantic communities** using frequency-domain descriptors (FFT), and models global interactions through **hierarchical community tokens** (HD-CoTAR). By restricting global attention to a small set of community tokens, it captures long-range dependencies beyond spatial proximity while keeping inference latency competitive with lightweight GNN baselines.

## Requirements

- Python 3.10+
- PyTorch
- `timm`, `numpy`, `pandas`, `scikit-learn`, `tqdm`

```bash
pip install -r requirements.txt
```

> Note: `requirements.txt` currently omits `tqdm`; install it manually (`pip install tqdm`) if needed.

## Datasets

Data for six benchmarks — PEMS03, PEMS04, PEMS07, PEMS08 (traffic flow), and METR-LA, PEMS-BAY (traffic speed) .

## Training

Run with a configuration file (one per dataset under `config/`):

```bash
python main.py --config config/PEMS04.conf
```

Key hyperparameters (input/output length, number of layers, learning rate, etc.) are set in each `.conf` file.

## Directory Structure

```
code/
├── main.py              # training & validation entry point (Solver)
├── models/
│   └── model.py         # FreqCoT, WindowAttBlock, HD-CoTAR module
├── lib/
│   └── utils.py         # data loading, metrics, logging
├── config/              # per-dataset .conf files
└── requirements.txt
```

## Citation

If you find this work useful, please cite our paper:

```
@article{freqcot,
  title   = {Frequency-enhanced Semantic Community Modeling for Spatiotemporal Forecasting},
  journal = {Under review},
  year    = {2026}
}
```
