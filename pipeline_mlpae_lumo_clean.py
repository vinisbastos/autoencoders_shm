
"""
pipeline_mlpae_lumo.py
"""

from __future__ import annotations

import argparse
import json
import random
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
import torch.optim as optim
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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
    preprocess: str = "CP4"
    sequence_length: int = 100
    overlap: int = 0
    bottleneck_size: int = 1200
    max_epochs: int = 5000
    learning_rate: float = 1e-3
    batch_size: int = 128
    patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    validation_size: float = 0.1
    seed: int = 42
    low_hz: float = 2.0
    high_hz: float = 80.0
    filter_order: int = 4
    output_dir: str = "outputs_mlp_ae"

class MLPAutoencoder(nn.Module):
    def __init__(self, input_size: int, bottleneck_size: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, bottleneck_size), nn.ReLU())
        self.decoder = nn.Linear(bottleneck_size, input_size)
    def forward(self, x):
        return self.decoder(self.encoder(x))

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdir(path) -> Path:
    path = Path(path); path.mkdir(parents=True, exist_ok=True); return path

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

def download_files(scenario: str, data_dir: Path) -> Tuple[Path, List[Path]]:
    if gdown is None:
        raise ImportError("Instale gdown: pip install gdown")
    mkdir(data_dir)
    cfg = SCENARIO_URLS[scenario]
    train_path = data_dir / f"{scenario}_Train.mat"
    test_paths = [data_dir / f"{scenario}_Test_{i}.mat" for i in range(1, 7)]
    if not train_path.exists(): gdown.download(cfg["train"], str(train_path), quiet=False)
    for url, path in zip(cfg["tests"], test_paths):
        if not path.exists(): gdown.download(url, str(path), quiet=False)
    return train_path, test_paths

def _decode_h5_string(obj) -> str:
    arr = np.array(obj)
    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(c)) for c in arr.ravel() if int(c) != 0).strip()
    return "".join(arr.astype(str).ravel()).strip()

def read_channel_names(f: h5py.File) -> List[str]:
    if "ChannelNames" not in f:
        return [f"channel_{i}" for i in range(np.array(f["Data"]).shape[0])]
    raw = np.array(f["ChannelNames"])
    if raw.dtype == object:
        return [_decode_h5_string(f[ref]) for ref in raw.ravel()]
    if raw.ndim == 2:
        return [_decode_h5_string(raw[i, :]) for i in range(raw.shape[0])]
    return [_decode_h5_string(f["ChannelNames"])]

def read_mat(path: Path) -> dict:
    """
    Lê um arquivo .mat da base LUMO.

    Os arquivos utilizados neste trabalho armazenam os dados dentro do grupo 'Dat',
    conforme a estrutura:
        Dat/Data
        Dat/ChannelNames
        Dat/Fs
        Dat/Time
        Dat/Timestamps
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        if "Dat" not in f:
            raise KeyError(
                f"O arquivo {path} não contém o grupo 'Dat'. "
                f"Chaves disponíveis no nível raiz: {list(f.keys())}"
            )

        dat_group = f["Dat"]

        # Decodificação dos nomes dos canais.
        channel_names = []
        for i in range(dat_group["ChannelNames"].shape[0]):
            name_ref = dat_group["ChannelNames"][i, 0]
            raw_name = f[name_ref][()]
            decoded_name = "".join(chr(char[0]) for char in raw_name)
            channel_names.append(decoded_name)

        data = np.array(dat_group["Data"])
        fs = float(np.array(dat_group["Fs"]).squeeze())

    return {
        "Data": data,
        "ChannelNames": channel_names,
        "Fs": fs,
    }


def load_data(scenario: str, data_dir: Path, download: bool):
    if download:
        train_path, test_paths = download_files(scenario, data_dir)
    else:
        train_path = data_dir / f"{scenario}_Train.mat"
        test_paths = [data_dir / f"{scenario}_Test_{i}.mat" for i in range(1, 7)]
        missing = [p for p in [train_path, *test_paths] if not p.exists()]
        if missing: raise FileNotFoundError(f"Arquivos ausentes: {missing}. Use --download.")
    return read_mat(train_path), [read_mat(p) for p in test_paths]

def accelerometers(ds: dict) -> Tuple[np.ndarray, List[str]]:
    names = list(ds["ChannelNames"])
    idx = [i for i, n in enumerate(names) if "accel" in str(n).lower()]
    if len(idx) != N_SENSORS: raise ValueError(f"Esperados 18 acelerômetros; encontrados {len(idx)}.")
    return np.asarray(ds["Data"])[idx, :], [names[i] for i in idx]

def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    b, a = signal.butter(order, [low/(0.5*fs), high/(0.5*fs)], btype="bandpass")
    return signal.filtfilt(b, a, x, axis=1)

def windows_flat(x: np.ndarray, sl: int, ov: int) -> np.ndarray:
    if not 0 <= ov < sl: raise ValueError("overlap deve ser 0 <= overlap < sequence_length")
    step = sl - ov
    out = [x[:, i:i+sl].reshape(-1) for i in range(0, x.shape[1]-sl+1, step)]
    if not out: raise ValueError("Nenhuma janela criada.")
    return np.asarray(out, dtype=np.float32)

def preprocess(train_x, test_xs, fs_train, fs_tests, cfg: Config):
    cp = cfg.preprocess.upper()
    if cp not in {"CP1","CP2","CP3","CP4"}: raise ValueError("CP inválido")
    if cp in {"CP2","CP4"}:
        train_x = bandpass(train_x, fs_train, cfg.low_hz, cfg.high_hz, cfg.filter_order)
        test_xs = [bandpass(x, fs, cfg.low_hz, cfg.high_hz, cfg.filter_order) for x, fs in zip(test_xs, fs_tests)]
    train_seq = windows_flat(train_x, cfg.sequence_length, cfg.overlap)
    test_seq = [windows_flat(x, cfg.sequence_length, cfg.overlap) for x in test_xs]
    if cp in {"CP3","CP4"}:
        scaler = RobustScaler()
        train_seq = scaler.fit_transform(train_seq).astype(np.float32)
        test_seq = [scaler.transform(x).astype(np.float32) for x in test_seq]
    return train_seq, test_seq

def flat_to_3d(x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    if d % N_SENSORS != 0: raise ValueError("Dimensão incompatível com 18 sensores.")
    return x.reshape(n, N_SENSORS, d // N_SENSORS)

def train(model, train_arr, val_arr, cfg: Config, dev: torch.device):
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
    mse = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(train_arr, dtype=torch.float32)), batch_size=cfg.batch_size, shuffle=True)
    val_t = torch.tensor(val_arr, dtype=torch.float32, device=dev)
    hist, best, best_loss, wait = [], None, np.inf, 0
    for epoch in range(1, cfg.max_epochs+1):
        model.train(); losses=[]
        for (batch,) in loader:
            batch = batch.to(dev); opt.zero_grad(); loss = mse(model(batch), batch); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): val_loss = float(mse(model(val_t), val_t).item())
        tr_loss = float(np.mean(losses)); sch.step(val_loss)
        hist.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": val_loss})
        if val_loss < best_loss - 1e-12:
            best_loss, wait = val_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= cfg.patience: break
    if best is not None: model.load_state_dict(best)
    return model, pd.DataFrame(hist)

@torch.no_grad()
def reconstruct(model, x: np.ndarray, dev: torch.device, batch_size=1024) -> np.ndarray:
    model.eval(); outs=[]
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    for (batch,) in loader: outs.append(model(batch.to(dev)).cpu().numpy())
    return np.vstack(outs)

def rcov_window_features(x3, e3):
    x = x3.astype(np.float64); e = e3.astype(np.float64)
    return np.abs(np.mean((x-x.mean(axis=2, keepdims=True))*(e-e.mean(axis=2, keepdims=True)), axis=2))

def sensors_to_levels(v):
    return np.asarray(v).reshape(*np.asarray(v).shape[:-1], LEVELS, SENSORS_PER_LEVEL).mean(axis=-1)

def rcov_level(model, seq, dev):
    rec = reconstruct(model, seq, dev); feat = rcov_window_features(flat_to_3d(seq), flat_to_3d(seq-rec))
    return sensors_to_levels(feat.mean(axis=0))

def r2_global(model, seq, dev):
    rec = reconstruct(model, seq, dev)
    return float(r2_score(seq.reshape(-1), rec.reshape(-1)))

def md_models(model, train_seq, dev):
    rec = reconstruct(model, train_seq, dev)
    feat = rcov_window_features(flat_to_3d(train_seq), flat_to_3d(train_seq-rec))
    cov_g = EmpiricalCovariance().fit(feat)
    cov_l = []
    for lvl in range(LEVELS):
        i = 2*lvl; cov_l.append(EmpiricalCovariance().fit(feat[:, i:i+2]))
    return cov_g, cov_l

def md_eval(model, seq, dev, cov_g, cov_l):
    rec = reconstruct(model, seq, dev)
    feat = rcov_window_features(flat_to_3d(seq), flat_to_3d(seq-rec))
    mdg = float(cov_g.mahalanobis(feat).mean())
    mdl = []
    for lvl, cov in enumerate(cov_l):
        i = 2*lvl; mdl.append(float(cov.mahalanobis(feat[:, i:i+2]).mean()))
    return mdg, np.asarray(mdl)

def evaluate(model, train_seq, test_seq, dev):
    names = ["Referencia_integra_treino", "Referencia_integra_validacao", *[f"Alteracao_estrutural_{i}" for i in range(1,6)]]
    seqs = [train_seq, *test_seq]
    cov_g, cov_l = md_models(model, train_seq, dev)
    rcov_ref = rcov_level(model, train_seq, dev)
    global_rows, md_rows, rcov_rows = [], [], []
    for name, seq in zip(names, seqs):
        mdg, mdl = md_eval(model, seq, dev, cov_g, cov_l)
        rc = rcov_level(model, seq, dev)
        global_rows.append({"dataset": name, "n_windows": len(seq), "r2": r2_global(model, seq, dev), "md2_global_mean": mdg})
        for lvl in range(LEVELS):
            level = f"NM{lvl+1}"
            md_rows.append({"dataset": name, "level": level, "md2_level_mean": mdl[lvl]})
            rcov_rows.append({"dataset": name, "level": level, "rcov": rc[lvl], "delta_rcov_vs_train": abs(rc[lvl]-rcov_ref[lvl])})
    return pd.DataFrame(global_rows), pd.DataFrame(md_rows), pd.DataFrame(rcov_rows)

def save_curve(hist, path):
    fig, ax = plt.subplots(figsize=(8,4)); ax.plot(hist.epoch, hist.train_mse, label="Treino"); ax.plot(hist.epoch, hist.val_mse, label="Validação")
    ax.set_xlabel("Época"); ax.set_ylabel("MSE"); ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def save_outputs(out, tag, cfg, model, hist, gdf, mdf, rdf):
    out = mkdir(out); (out/f"{tag}_config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    hist.to_csv(out/f"{tag}_training_history.csv", index=False); gdf.to_csv(out/f"{tag}_global_metrics.csv", index=False)
    mdf.to_csv(out/f"{tag}_md2_by_level.csv", index=False); rdf.to_csv(out/f"{tag}_rcov_by_level.csv", index=False)
    torch.save(model.state_dict(), out/f"{tag}_model_state_dict.pt"); save_curve(hist, out/f"{tag}_training_curve.png")

def prepare(cfg, data_dir, download):
    train_ds, test_ds = load_data(cfg.scenario, data_dir, download)
    tr, _ = accelerometers(train_ds); tests = [accelerometers(ds)[0] for ds in test_ds]
    return preprocess(tr, tests, float(train_ds["Fs"]), [float(ds["Fs"]) for ds in test_ds], cfg)

def run_one(cfg: Config, data_dir: Path, download: bool, tag: str):
    set_seed(cfg.seed); dev = device()
    train_seq, test_seq = prepare(cfg, data_dir, download)
    tr, val = train_test_split(train_seq, test_size=cfg.validation_size, random_state=cfg.seed, shuffle=True)
    model = MLPAutoencoder(train_seq.shape[1], cfg.bottleneck_size); model.apply(init_weights)
    model, hist = train(model, tr, val, cfg, dev)
    gdf, mdf, rdf = evaluate(model, train_seq, test_seq, dev)
    out = Path(cfg.output_dir)/cfg.scenario/tag
    save_outputs(out, tag, cfg, model, hist, gdf, mdf, rdf)
    print(f"OK: {tag} -> {out}")

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline MLP-AE LUMO")
    p.add_argument("--scenario", default="DAM4_1", choices=list(SCENARIO_URLS.keys()))
    p.add_argument("--mode", default="reference_cp", choices=["reference_cp", "variants", "single"])
    p.add_argument("--preprocess", default="CP4", choices=["CP1","CP2","CP3","CP4"])
    p.add_argument("--sequence-length", type=int, default=100); p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--bottleneck-size", type=int, default=1200); p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--data-dir", default="data"); p.add_argument("--output-dir", default="outputs_mlp_ae")
    p.add_argument("--download", action="store_true")
    return p.parse_args()

def main():
    a = parse_args(); data_dir = Path(a.data_dir)
    if a.mode == "reference_cp":
        for cp in ["CP1","CP2","CP3","CP4"]:
            cfg = Config(a.scenario, cp, 100, 0, 1200, max_epochs=a.max_epochs, seed=a.seed, output_dir=a.output_dir)
            run_one(cfg, data_dir, a.download, f"reference_SL100_OV0_DZ1200_{cp}")
    elif a.mode == "variants":
        for dz in [1200, 12228]:
            cfg = Config(a.scenario, a.preprocess, 1024, 896, dz, max_epochs=a.max_epochs, seed=a.seed, output_dir=a.output_dir)
            run_one(cfg, data_dir, a.download, f"variant_SL1024_OV896_DZ{dz}_{a.preprocess}")
    else:
        cfg = Config(a.scenario, a.preprocess, a.sequence_length, a.overlap, a.bottleneck_size, max_epochs=a.max_epochs, seed=a.seed, output_dir=a.output_dir)
        run_one(cfg, data_dir, a.download, f"single_SL{a.sequence_length}_OV{a.overlap}_DZ{a.bottleneck_size}_{a.preprocess}")

if __name__ == "__main__":
    main()
