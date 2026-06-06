#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_cae1d_lumo_clean.py

Escopo:
- Modelo CAE-1D profundo sem conexões residuais/skip;
- Entrada multicanal no formato (N, 18, T), preservando sensores e tempo;
- Pré-processamento CP4: filtro Butterworth passa-banda + RobustScaler 3D por sensor;

"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    import gdown
except ImportError:
    gdown = None

LEVELS = 9
SENSORS_PER_LEVEL = 2
N_SENSORS = LEVELS * SENSORS_PER_LEVEL
DEFAULT_FS = 1651.6
EPS = 1e-12

SCENARIO_URLS: Dict[str, Dict[str, object]] = {
    "DAM6_3": {
        "train": "https://drive.google.com/uc?id=1BZ5S4Y0cCn_LXUvJHiCXxq85tKjDum20",
        "tests": [
            "https://drive.google.com/uc?id=1bIYrJSsy_PJkwc8sm5V_XM1vyyVtms5s",
            "https://drive.google.com/uc?id=1rpAWKQ1KE314eMXgAEFu9rOEseGLGzyf",
            "https://drive.google.com/uc?id=1Jr_jiMa7rqlhzfCnX0e0-ecVAsPx5Brz",
            "https://drive.google.com/uc?id=1aMB7ObOyqFO6MG8T2PErHWWIeBWV7CVd",
            "https://drive.google.com/uc?id=1g6xMwikrFb5GND3Ih7WDKwXVc9rMzpKe",
            "https://drive.google.com/uc?id=1V41G15UZ_oVMCvWEJ4uTBwJg7OTeOuEk",
        ],
    },
    "DAM4_3": {
        "train": "https://drive.google.com/uc?id=1AgrSFTVpqJlbr6tgBaiL2J0h_i9V2vvx",
        "tests": [
            "https://drive.google.com/uc?id=18oHg7ons7YwnqMW5qeeGv1GbDm2Lr-SL",
            "https://drive.google.com/uc?id=1Ndo-DmtW74NfI4lCAq5hcHasNeLGG3t8",
            "https://drive.google.com/uc?id=1dVN8n5_XwVgGvNKyeG6FyHGXGn-o4T89",
            "https://drive.google.com/uc?id=1glCu4OpMBYDqf5Wk5XjVH37inwzNdJPl",
            "https://drive.google.com/uc?id=1P1je1K3lS9BvXHo6FnR7E5pipCwF1jew",
            "https://drive.google.com/uc?id=13Mu9pJW3nZP9_4wSUkSJ9YtmqhvYYD_n",
        ],
    },
    "DAM3_3": {
        "train": "https://drive.google.com/uc?id=1vFvNvi-yV-tJc3xRMH1O0yL8CDqUuTR2",
        "tests": [
            "https://drive.google.com/uc?id=1r85rucSb-ygUdT2Ggq0dncuO8ugt5tGQ",
            "https://drive.google.com/uc?id=1TdaI79CfI2l9or_cCXjnlOln5dfso_Sh",
            "https://drive.google.com/uc?id=1ACYyFZnwIAwke3lDsUrQPEyQGgTU2U7W",
            "https://drive.google.com/uc?id=1KVXy2yZmDCMgnR0L4i3zheYSK2TaY6HX",
            "https://drive.google.com/uc?id=1zEqXBI517p9mUUqHAIFkRDTIdttj05_K",
            "https://drive.google.com/uc?id=1RwT_aUZs3sr6qfO7dKPItUYnIvA4PyvY",
        ],
    },
    "DAM4_1": {
        "train": "https://drive.google.com/uc?id=1mnGtnMzTEEnWPvIlOQY69qv1cLDl-Z3T",
        "tests": [
            "https://drive.google.com/uc?id=1beH4eKoeEyyajr1rxQzO7uybxp5rJxZp",
            "https://drive.google.com/uc?id=1vS0y-fjN6lVpQD5EevuLNFuSm_lZspVU",
            "https://drive.google.com/uc?id=1RG4HPdk2ohCrdK5-D7jTD95eaWsK0oD8",
            "https://drive.google.com/uc?id=1Kqr_rlNnI148OHSVd2Bn-PgA8K1ppHM6",
            "https://drive.google.com/uc?id=1HqZK4K6PEQeP60L93tlYDkm6lwvW8LYm",
            "https://drive.google.com/uc?id=1HqZK4K6PEQeP60L93tlYDkm6lwvW8LYm",
        ],
    },
}


@dataclass
class Config:
    scenario: str = "DAM4_1"
    sequence_length: int = 1024
    overlap: int = 896
    max_epochs: int = 3500
    log_every: int = 10
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    patience: int = 40
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    validation_size: float = 0.1
    seed: int = 42
    low_hz: float = 2.0
    high_hz: float = 80.0
    filter_order: int = 4
    grad_clip_norm: float = 1.0
    n_sensors: int = 18
    c1: int = 64
    c2: int = 128
    c3: int = 256
    latent_c: int = 64
    kernel_size: int = 5
    use_gap: bool = False
    output_dir: str = "outputs_cae1d"


# =============================================================================
# Utilitários gerais
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mkdir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Download e leitura dos arquivos MAT
# =============================================================================


def download_files(scenario: str, data_dir: Path) -> Tuple[Path, List[Path]]:
    if gdown is None:
        raise ImportError("Instale gdown: pip install gdown")
    mkdir(data_dir)
    cfg = SCENARIO_URLS[scenario]
    train_path = data_dir / f"{scenario}_Train.mat"
    test_paths = [data_dir / f"{scenario}_Test_{i}.mat" for i in range(1, 7)]
    if not train_path.exists():
        gdown.download(cfg["train"], str(train_path), quiet=False)
    for url, path in zip(cfg["tests"], test_paths):
        if not path.exists():
            gdown.download(url, str(path), quiet=False)
    return train_path, test_paths


def read_mat(path: Path) -> dict:
    """Lê arquivo .mat da base LUMO armazenado no grupo HDF5 'Dat'."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "Dat" not in f:
            raise KeyError(f"O arquivo {path} não contém o grupo 'Dat'. Chaves: {list(f.keys())}")
        dat_group = f["Dat"]
        channel_names = []
        for i in range(dat_group["ChannelNames"].shape[0]):
            name_ref = dat_group["ChannelNames"][i, 0]
            raw_name = f[name_ref][()]
            decoded_name = "".join(chr(char[0]) for char in raw_name)
            channel_names.append(decoded_name)
        data = np.array(dat_group["Data"])
        fs = float(np.array(dat_group["Fs"]).squeeze())
    return {"Data": data, "ChannelNames": channel_names, "Fs": fs}


def load_data(scenario: str, data_dir: Path, download: bool):
    if download:
        train_path, test_paths = download_files(scenario, data_dir)
    else:
        train_path = data_dir / f"{scenario}_Train.mat"
        test_paths = [data_dir / f"{scenario}_Test_{i}.mat" for i in range(1, 7)]
        missing = [p for p in [train_path, *test_paths] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Arquivos ausentes: {missing}. Use --download.")
    return read_mat(train_path), [read_mat(p) for p in test_paths]


def accelerometers(ds: dict) -> Tuple[np.ndarray, List[str]]:
    names = list(ds["ChannelNames"])
    idx = [i for i, n in enumerate(names) if "accel" in str(n).lower()]
    if len(idx) != N_SENSORS:
        raise ValueError(f"Esperados 18 acelerômetros; encontrados {len(idx)}.")
    return np.asarray(ds["Data"])[idx, :], [names[i] for i in idx]


# =============================================================================
# Pré-processamento CP4 e segmentação 3D
# =============================================================================


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    b, a = signal.butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass")
    return signal.filtfilt(b, a, x, axis=1)


def windows_3d(x: np.ndarray, sl: int, ov: int) -> np.ndarray:
    """Cria janelas no formato (N, 18, SL), sem janela parcial no final."""
    if not 0 <= ov < sl:
        raise ValueError("overlap deve ser 0 <= overlap < sequence_length")
    step = sl - ov
    n_sensors, n_samples = x.shape
    starts = range(0, n_samples - sl + 1, step)
    out = [x[:, start:start + sl] for start in starts]
    if not out:
        raise ValueError("Nenhuma janela criada.")
    return np.asarray(out, dtype=np.float32)


def fit_robust_scaler_3d(train_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ajusta mediana e IQR por sensor usando treino íntegro."""
    q50 = np.median(train_seq, axis=(0, 2))
    q25 = np.percentile(train_seq, 25, axis=(0, 2))
    q75 = np.percentile(train_seq, 75, axis=(0, 2))
    iqr = np.maximum(q75 - q25, 1e-8)
    return q50.astype(np.float32), iqr.astype(np.float32)


def apply_robust_scaler_3d(seq: np.ndarray, q50: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return ((seq - q50[None, :, None]) / iqr[None, :, None]).astype(np.float32)


def preprocess(train_x, test_xs, fs_train, fs_tests, cfg: Config):
    """Aplica CP4: filtro passa-banda + robust scaling 3D por sensor."""
    train_x = bandpass(train_x, fs_train, cfg.low_hz, cfg.high_hz, cfg.filter_order)
    test_xs = [bandpass(x, fs, cfg.low_hz, cfg.high_hz, cfg.filter_order) for x, fs in zip(test_xs, fs_tests)]
    train_seq = windows_3d(train_x, cfg.sequence_length, cfg.overlap)
    test_seq = [windows_3d(x, cfg.sequence_length, cfg.overlap) for x in test_xs]
    q50, iqr = fit_robust_scaler_3d(train_seq)
    train_seq = apply_robust_scaler_3d(train_seq, q50, iqr)
    test_seq = [apply_robust_scaler_3d(x, q50, iqr) for x in test_seq]
    return train_seq, test_seq


# =============================================================================
# Modelo CAE-1D
# =============================================================================


def group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
    return nn.GroupNorm(num_groups=1, num_channels=channels)


class ConvBlock1D(nn.Module):
    """Conv1d -> GroupNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            group_norm(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CAE1DNoSkip(nn.Module):
    """Autoencoder convolucional 1D sem skip/residual e sem dilatação."""

    def __init__(
        self,
        n_sensors: int = 18,
        c1: int = 64,
        c2: int = 128,
        c3: int = 256,
        latent_c: int = 64,
        kernel_size: int = 5,
        use_gap: bool = False,
    ):
        super().__init__()
        self.use_gap = use_gap
        k = kernel_size
        self.enc1 = nn.Sequential(ConvBlock1D(n_sensors, c1, k, 1), ConvBlock1D(c1, c1, k, 1))
        self.down1 = ConvBlock1D(c1, c2, k, 2)
        self.enc2 = nn.Sequential(ConvBlock1D(c2, c2, k, 1), ConvBlock1D(c2, c2, k, 1))
        self.down2 = ConvBlock1D(c2, c3, k, 2)

        if self.use_gap:
            self.to_latent = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(c3, latent_c, 1), nn.ReLU())
            self.from_latent = nn.Sequential(nn.Conv1d(latent_c, c3, 1), nn.ReLU())
        else:
            self.to_latent = nn.Sequential(nn.Conv1d(c3, latent_c, 1), group_norm(latent_c), nn.ReLU())
            self.from_latent = nn.Sequential(nn.Conv1d(latent_c, c3, 1), group_norm(c3), nn.ReLU())

        self.up1 = nn.Sequential(nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1), group_norm(c2), nn.ReLU())
        self.dec1 = nn.Sequential(ConvBlock1D(c2, c2, k, 1), ConvBlock1D(c2, c2, k, 1))
        self.up2 = nn.Sequential(nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1), group_norm(c1), nn.ReLU())
        self.dec2 = nn.Sequential(ConvBlock1D(c1, c1, k, 1), ConvBlock1D(c1, c1, k, 1))
        self.out = nn.Conv1d(c1, n_sensors, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        length_in = x.size(-1)
        h = self.enc1(x)
        h = self.down1(h)
        h = self.enc2(h)
        h = self.down2(h)
        z = self.to_latent(h)
        if self.use_gap:
            target = max(1, length_in // 4)
            h = self.from_latent(z).expand(-1, -1, target)
        else:
            h = self.from_latent(z)
        y = self.up1(h)
        y = self.dec1(y)
        y = self.up2(y)
        y = self.dec2(y)
        y = self.out(y)
        if y.size(-1) != length_in:
            y = F.interpolate(y, size=length_in, mode="linear", align_corners=False)
        z_vec = z.mean(dim=-1)
        return y, z_vec


def init_weights(module: nn.Module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# Treinamento e inferência
# =============================================================================


def train(model, train_arr, val_arr, cfg: Config, dev: torch.device):
    model.to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.scheduler_patience, factor=cfg.scheduler_factor
    )
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(train_arr, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_tensor = torch.tensor(val_arr, dtype=torch.float32, device=dev)
    hist, best, best_loss, wait = [], None, np.inf, 0
    start_time = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        batch_losses = []
        for (batch,) in loader:
            batch = batch.to(dev)
            y, _ = model(batch)
            loss = criterion(y, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for j in range(0, val_tensor.size(0), cfg.batch_size):
                xb = val_tensor[j:j + cfg.batch_size]
                yb, _ = model(xb)
                val_losses.append(float(criterion(yb, xb).item()))
        train_loss = float(np.mean(batch_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        hist.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        if val_loss < best_loss - 1e-12:
            best_loss, wait = val_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch == 1 or epoch % cfg.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch:04d}/{cfg.max_epochs}] "
                f"train_mse={train_loss:.6e} | val_mse={val_loss:.6e} | "
                f"best_val={best_loss:.6e} | wait={wait}/{cfg.patience} | "
                f"elapsed={int(elapsed//60):02d}:{int(elapsed%60):02d}",
                flush=True,
            )

        if wait >= cfg.patience:
            print(f"Early stopping acionado na época {epoch}. Melhor val_mse={best_loss:.6e}", flush=True)
            break

    if best is not None:
        model.load_state_dict(best)
    return model, pd.DataFrame(hist)


@torch.no_grad()
def reconstruct(model, x: np.ndarray, dev: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    outs = []
    for (batch,) in loader:
        y, _ = model(batch.to(dev))
        outs.append(y.cpu().numpy())
    return np.vstack(outs)


# =============================================================================
# Métricas RCOV, MD² e R²
# =============================================================================


def rcov_window_features(x3, e3):
    x = x3.astype(np.float64)
    e = e3.astype(np.float64)
    return np.abs(np.mean((x - x.mean(axis=2, keepdims=True)) * (e - e.mean(axis=2, keepdims=True)), axis=2))


def sensors_to_levels(v):
    return np.asarray(v).reshape(*np.asarray(v).shape[:-1], LEVELS, SENSORS_PER_LEVEL).mean(axis=-1)


def rcov_level(model, seq, dev):
    rec = reconstruct(model, seq, dev)
    feat = rcov_window_features(seq, seq - rec)
    return sensors_to_levels(feat.mean(axis=0))


def r2_global(model, seq, dev):
    rec = reconstruct(model, seq, dev)
    return float(r2_score(seq.reshape(-1), rec.reshape(-1)))


def md_models(model, train_seq, dev):
    rec = reconstruct(model, train_seq, dev)
    feat = rcov_window_features(train_seq, train_seq - rec)
    cov_g = EmpiricalCovariance().fit(feat)
    cov_l = []
    for lvl in range(LEVELS):
        i = 2 * lvl
        cov_l.append(EmpiricalCovariance().fit(feat[:, i:i + 2]))
    return cov_g, cov_l


def md_eval(model, seq, dev, cov_g, cov_l):
    rec = reconstruct(model, seq, dev)
    feat = rcov_window_features(seq, seq - rec)
    mdg = float(cov_g.mahalanobis(feat).mean())
    mdl = []
    for lvl, cov in enumerate(cov_l):
        i = 2 * lvl
        mdl.append(float(cov.mahalanobis(feat[:, i:i + 2]).mean()))
    return mdg, np.asarray(mdl)


def evaluate(model, train_seq, test_seq, dev):
    names = ["Referencia_integra_treino", "Referencia_integra_validacao", *[f"Alteracao_estrutural_{i}" for i in range(1, 6)]]
    seqs = [train_seq, *test_seq]
    cov_g, cov_l = md_models(model, train_seq, dev)
    rcov_ref = rcov_level(model, train_seq, dev)
    global_rows, md_rows, rcov_rows = [], [], []
    for name, seq in zip(names, seqs):
        mdg, mdl = md_eval(model, seq, dev, cov_g, cov_l)
        rc = rcov_level(model, seq, dev)
        global_rows.append({"dataset": name, "n_windows": len(seq), "r2": r2_global(model, seq, dev), "md2_global_mean": mdg})
        for lvl in range(LEVELS):
            level = f"NM{lvl + 1}"
            md_rows.append({"dataset": name, "level": level, "md2_level_mean": mdl[lvl]})
            rcov_rows.append({"dataset": name, "level": level, "rcov": rc[lvl], "delta_rcov_vs_train": abs(rc[lvl] - rcov_ref[lvl])})
    return pd.DataFrame(global_rows), pd.DataFrame(md_rows), pd.DataFrame(rcov_rows)


# =============================================================================
# Saídas
# =============================================================================


def save_curve(hist, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.epoch, hist.train_mse, label="Treino")
    ax.plot(hist.epoch, hist.val_mse, label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_outputs(out, tag, cfg, model, hist, gdf, mdf, rdf):
    out = mkdir(out)
    (out / f"{tag}_config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    hist.to_csv(out / f"{tag}_training_history.csv", index=False)
    gdf.to_csv(out / f"{tag}_global_metrics.csv", index=False)
    mdf.to_csv(out / f"{tag}_md2_by_level.csv", index=False)
    rdf.to_csv(out / f"{tag}_rcov_by_level.csv", index=False)
    torch.save(model.state_dict(), out / f"{tag}_model_state_dict.pt")
    save_curve(hist, out / f"{tag}_training_curve.png")


# =============================================================================
# Execução
# =============================================================================


def prepare(cfg, data_dir, download):
    train_ds, test_ds = load_data(cfg.scenario, data_dir, download)
    tr, _ = accelerometers(train_ds)
    tests = [accelerometers(ds)[0] for ds in test_ds]
    return preprocess(tr, tests, float(train_ds["Fs"]), [float(ds["Fs"]) for ds in test_ds], cfg)


def run_one(cfg: Config, data_dir: Path, download: bool, tag: str):
    set_seed(cfg.seed)
    dev = device()
    print("=" * 80, flush=True)
    print(f"Iniciando experimento: {tag}", flush=True)
    print(f"Arquitetura: CAE-1D", flush=True)
    print(f"Cenário: {cfg.scenario}", flush=True)
    print(f"Pré-processamento: CP4 (filtro {cfg.low_hz:g}-{cfg.high_hz:g} Hz + RobustScaler 3D)", flush=True)
    print(f"SL={cfg.sequence_length}, OV={cfg.overlap}", flush=True)
    print(f"Canais: c1={cfg.c1}, c2={cfg.c2}, c3={cfg.c3}, latent_c={cfg.latent_c}", flush=True)
    print(f"Dispositivo: {dev}", flush=True)
    print("=" * 80, flush=True)

    print("[1/5] Carregando dados e aplicando pré-processamento...", flush=True)
    train_seq, test_seq = prepare(cfg, data_dir, download)
    print(f"Janelas de treino: {train_seq.shape}", flush=True)
    for i, seq in enumerate(test_seq, start=1):
        print(f"Janelas de teste {i}: {seq.shape}", flush=True)

    print("[2/5] Separando treino e validação...", flush=True)
    tr, val = train_test_split(train_seq, test_size=cfg.validation_size, random_state=cfg.seed, shuffle=True)
    print(f"Treino: {tr.shape} | Validação: {val.shape}", flush=True)

    print("[3/5] Inicializando modelo CAE-1D...", flush=True)
    model = CAE1DNoSkip(
        n_sensors=cfg.n_sensors,
        c1=cfg.c1,
        c2=cfg.c2,
        c3=cfg.c3,
        latent_c=cfg.latent_c,
        kernel_size=cfg.kernel_size,
        use_gap=cfg.use_gap,
    )
    model.apply(init_weights)

    print("[4/5] Treinando modelo...", flush=True)
    model, hist = train(model, tr, val, cfg, dev)

    print("[5/5] Calculando métricas e salvando resultados...", flush=True)
    gdf, mdf, rdf = evaluate(model, train_seq, test_seq, dev)
    out = Path(cfg.output_dir) / cfg.scenario / tag
    save_outputs(out, tag, cfg, model, hist, gdf, mdf, rdf)
    print(f"Experimento concluído. Resultados salvos em: {out}", flush=True)
    print("=" * 80, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline CAE-1D para a base LUMO")
    p.add_argument("--scenario", default="DAM4_1", choices=list(SCENARIO_URLS.keys()))
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=896)
    p.add_argument("--max-epochs", type=int, default=3500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--low-hz", type=float, default=2.0)
    p.add_argument("--high-hz", type=float, default=80.0)
    p.add_argument("--c1", type=int, default=64)
    p.add_argument("--c2", type=int, default=128)
    p.add_argument("--c3", type=int, default=256)
    p.add_argument("--latent-c", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--use-gap", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs_cae1d")
    p.add_argument("--download", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config(
        scenario=a.scenario,
        sequence_length=a.sequence_length,
        overlap=a.overlap,
        max_epochs=a.max_epochs,
        log_every=a.log_every,
        learning_rate=a.learning_rate,
        batch_size=a.batch_size,
        patience=a.patience,
        low_hz=a.low_hz,
        high_hz=a.high_hz,
        c1=a.c1,
        c2=a.c2,
        c3=a.c3,
        latent_c=a.latent_c,
        kernel_size=a.kernel_size,
        use_gap=a.use_gap,
        seed=a.seed,
        output_dir=a.output_dir,
    )
    tag = f"CAE1D_SL{cfg.sequence_length}_OV{cfg.overlap}_CP4_{int(cfg.low_hz)}-{int(cfg.high_hz)}Hz"
    run_one(cfg, Path(a.data_dir), a.download, tag)


if __name__ == "__main__":
    main()
