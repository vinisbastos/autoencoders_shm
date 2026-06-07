# Autoencoder Pipelines — LUMO SHM

This repository contains clean and reproducible versions of the pipelines used in the dissertation for detection and localization of structural changes in the LUMO structure, based on acceleration signals and autoencoder models trained exclusively using data from the healthy structural reference condition.

Currently, the repository includes three architectures:

- **MLP-AE**: dense-layer autoencoder with flattened input.
- **CAE-1D**: one-dimensional convolutional autoencoder with multichannel input in the format `(N, 18, T)`.
- **TCN-AE**: Temporal Convolutional Network-based autoencoder, using dilated convolutional blocks and multichannel input in the format `(N, 18, T)`.

The metrics exported by the pipelines include:

- **RCOV** per measurement level;
- **quadratic Mahalanobis distance**, `MD²`, both global and per level;
- **global coefficient of determination**, `R²`.

Please cite this work as "Bastos, Vinicius Silveira. Aplicação de modelos de aprendizado de máquina para detecção de danos estruturais em um mastro treliçado de benchmark. 2026. 107 f. Dissertação (Mestrado Profissional em Instrumentação, Controle e Automação de Processos de Mineração) - Escola de Minas, Universidade Federal de Ouro Preto, Ouro Preto, 2025."

---

## 1. Repository structure

```text
.
├── pipeline_mlpae_lumo.py
├── pipeline_cae1d_lumo.py
├── pipeline_tcnae_lumo.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── DAM6_3_Train.mat
    ├── DAM6_3_Test_1.mat
    ├── ...
    └── DAM6_3_Test_6.mat
```

The `data/` folder is used locally to store `.mat` files from the LUMO dataset. This folder should not be versioned on GitHub.

---

## 2. Data

The scripts can automatically download the `.mat` files from the LUMO dataset using `gdown`, through the option:

```bash
--download
```

If the files are downloaded or organized manually, they must be stored in the `data/` directory using the following naming convention:

```text
data/<scenario>_Train.mat
data/<scenario>_Test_1.mat
data/<scenario>_Test_2.mat
data/<scenario>_Test_3.mat
data/<scenario>_Test_4.mat
data/<scenario>_Test_5.mat
data/<scenario>_Test_6.mat
```

Example for scenario `DAM6_3`:

```text
data/DAM6_3_Train.mat
data/DAM6_3_Test_1.mat
data/DAM6_3_Test_2.mat
data/DAM6_3_Test_3.mat
data/DAM6_3_Test_4.mat
data/DAM6_3_Test_5.mat
data/DAM6_3_Test_6.mat
```

Available scenarios:

```text
DAM6_3
DAM4_3
DAM3_3
DAM4_1
```

---

## 3. MLP-AE Pipeline

The file `pipeline_mlpae_lumo.py` implements the dense autoencoder used in the dissertation.

In this model, each multichannel temporal window is flattened before being fed into the network. Thus, a window originally structured as:

```text
18 sensors × SL samples
```

is converted into a vector of dimension:

```text
18 × SL
```

Examples:

```text
SL = 100   → input with 1800 features
SL = 1024  → input with 18432 features
```

### 3.1 MLP-AE specifics

The MLP-AE pipeline consists of two main stages:

1. **Reference pipeline** for evaluating preprocessing scenarios.
2. **Execution of final variants** using long windows.

---

### 3.2 Reference MLP-AE

The reference MLP-AE is used to evaluate preprocessing scenarios CP1, CP2, CP3, and CP4.

Default configuration:

```text
Architecture: MLP-AE
SL = 100
OV = 0
DZ = 1200
learning_rate = 1e-3
batch_size = 128
max_epochs = 5000
patience = 10
scheduler_patience = 5
scheduler_factor = 0.5
```

Preprocessing scenarios:

```text
CP1: raw signal
CP2: band-pass filter
CP3: RobustScaler
CP4: band-pass filter + RobustScaler
```

---

### 3.3 MLP-AE variants

Final MLP-AE variants use long windows with overlap:

```text
SL = 1024
OV = 896
```

Variants evaluated:

```text
MLP-AE-SL1024-DZ1200
MLP-AE-SL1024-DZ12228
```

---

### 3.4 Getting started — MLP-AE

Quick test of the reference MLP-AE:

```bash
python pipeline_mlpae_lumo.py --scenario DAM6_3 --mode single --preprocess CP4 --sequence-length 100 --overlap 0 --bottleneck-size 1200 --max-epochs 2 --log-every 1 --download
```

Evaluate CP1–CP4:

```bash
python pipeline_mlpae_lumo.py --scenario DAM6_3 --mode reference_cp --download
```

---

### 3.5 MLP-AE outputs

Results are stored in:

```text
outputs_mlp_ae/<scenario>/<experiment>/
```

---

## 4. CAE-1D Pipeline

The file `pipeline_cae1d_lumo.py` implements the one-dimensional convolutional autoencoder.

Unlike the MLP-AE, CAE-1D preserves multichannel structure:

```text
N × 18 × SL
```

---

### 4.1 CAE-1D specifics

- Temporal 1D convolutions  
- Channels represent sensors  
- No data flattening  

---

### 4.2 Preprocessing

Fixed CP4:

```text
band-pass filter + 3D RobustScaler per sensor
```

---

### 4.3 Architecture

```text
Conv1d
GroupNorm
ReLU
ConvTranspose1d
```

---

### 4.4 Examples

```bash
python pipeline_cae1d_lumo.py --scenario DAM6_3 --max-epochs 2 --download
```

---

### 4.5 CAE-1D outputs

```text
outputs_cae1d/<scenario>/<experiment>/
```

---

## 5. TCN-AE Pipeline

The file `pipeline_tcnae_lumo.py` implements the Temporal Convolutional Network autoencoder.

Input format:

```text
N × 18 × SL
```

---

### 5.1 TCN-AE specifics

- Dilated temporal convolutions  
- Residual connections  
- Large receptive field without pooling  

---

### 5.2 Preprocessing

Fixed CP4:

```text
band-pass filter + 3D RobustScaler per sensor
```

---

### 5.3 Architecture

```text
Input Conv1d
Dilated TCN blocks
Bottleneck
Decoder TCN blocks
Output Conv1d
```

---

### 5.4 Variants

```text
TCN5: (1, 2, 4, 8, 16)
TCN9: (1, 2, 4, 8, 16, 32, 64, 128, 256)
```

---

### 5.5 Examples

```bash
python pipeline_tcnae_lumo.py --scenario DAM6_3 --variant TCN5 --download
```

---

### 5.6 Outputs

```text
outputs_tcnae/<scenario>/<experiment>/
```

---

## 6. Exported files

All pipelines export:

```text
*_config.json
*_training_history.csv
*_global_metrics.csv
*_md2_by_level.csv
*_rcov_by_level.csv
*_model_state_dict.pt
*_training_curve.png
```

Descriptions:

- `*_config.json`: experiment configuration
- `*_training_history.csv`: training history
- `*_global_metrics.csv`: global metrics (`R²`, MD²)
- `*_md2_by_level.csv`: MD² per measurement level
- `*_rcov_by_level.csv`: RCOV per level
- `*_model_state_dict.pt`: trained model weights
- `*_training_curve.png`: loss curves

---