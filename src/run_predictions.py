#!/usr/bin/env python3
"""
predict_hierarchical.py
=======================
Prédiction hiérarchique IDP sans accès aux labels.

  Task 1 (MLP, flat)   : structure (0) vs disorder (1)
  Task 2 (GRU, window) : disorder (1) vs disorder-binding (2)

Les deux tasks utilisent des embeddings distogram (entropie de Shannon, L×25)
mais depuis deux fichiers H5 distincts :
  --h5_task1  →  H5 utilisé à l'entraînement de task1 (MLP)
  --h5_task2  →  H5 utilisé à l'entraînement de task2 (GRU)

Structure H5 attendue (même format pour les deux) :
  /proteins/<uid>/distogram_logits  [N_runs, L, L, 64]
  (clé 'labels' non requise)

Pipeline par résidu :
  score_disorder = sigmoid(MLP(emb_t1_flat))
  si score_disorder >= thr_t1  → désordonné
      score_binding = sigmoid(GRU(emb_t2_window))
      si score_binding >= thr_t2 → classe 2 (disorder-binding)
      sinon                       → classe 1 (disorder)
  sinon → classe 0 (structure)

Sortie CAID : un fichier TSV par protéine (intersection des UIDs des deux H5)
  residue_index  score_disorder  score_binding  predicted_class

Usage :
  python predict_hierarchical.py \
      --model_task1  /opt/models/task1_MLP_optimized_MLP.pt \
      --model_task2  /opt/models/task2_GRU_optimized_GRU.pt \
      --h5_task1     embeddings_task1.h5 \
      --h5_task2     embeddings_task2.h5 \
      --out_dir      predictions/ \
      [--threshold_task1 0.5] [--threshold_task2 0.5] [--window 64]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISTOGRAM_N_RUNS = 25   # nombre de runs d'entropie conservés (identique au training)
WINDOW_SIZE      = 64   # fenêtre GRU task2
BATCH_SIZE       = 1024


# ──────────────────────────────────────────────────────────────────────────────
# ARCHITECTURES  (identiques à retrain_optimize.py)
# ──────────────────────────────────────────────────────────────────────────────

class MLPNet(nn.Module):
    def __init__(self, d: int, hidden1: int = 256, hidden2: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden1), nn.LayerNorm(hidden1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),                  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 32),                       nn.GELU(),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, d: int, hidden: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(d, hidden, num_layers=num_layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, out.size(1) // 2, :]).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# INFÉRENCE DE L'ARCHITECTURE DEPUIS LE STATE_DICT
# ──────────────────────────────────────────────────────────────────────────────

def _infer_mlp_arch(sd: dict) -> dict:
    """
    net.0.weight : (hidden1, feat_dim)
    net.4.weight : (hidden2, hidden1)
    """
    p = {}
    w0 = sd.get("net.0.weight")
    w4 = sd.get("net.4.weight")
    if w0 is not None:
        p["feat_dim"] = int(w0.shape[1])
        p["hidden1"]  = int(w0.shape[0])
    if w4 is not None:
        p["hidden2"]  = int(w4.shape[0])
    return p


def _infer_gru_arch(sd: dict) -> dict:
    """
    gru.weight_ih_l0 : (3*hidden, feat_dim)   ← GRU a 3 gates (r, z, n), pas 4
    une clé weight_ih_l<i> (sans _reverse) par layer
    """
    p = {}
    w = sd.get("gru.weight_ih_l0")
    if w is not None:
        p["feat_dim"] = int(w.shape[1])
        p["hidden"]   = int(w.shape[0]) // 3   # ← 3 gates GRU, pas 4 (LSTM)
    # Compter uniquement les clés forward (exclure *_reverse)
    n_layers = sum(
        1 for k in sd
        if k.startswith("gru.weight_ih_l") and "_reverse" not in k
    )
    if n_layers > 0:
        p["num_layers"] = n_layers
    return p


# ──────────────────────────────────────────────────────────────────────────────
# SCALER LÉGER (reconstruit depuis mean_ / scale_ sauvegardés dans le .pt)
# ──────────────────────────────────────────────────────────────────────────────

class _Scaler:
    """Reproduit StandardScaler.transform() sans dépendance sklearn."""
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_  = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(np.float32)


def _load_scaler(payload: dict):
    """Retourne un _Scaler ou None si les clés sont absentes."""
    mean  = payload.get("scaler_mean")
    scale = payload.get("scaler_scale")
    if mean is None or scale is None:
        return None
    return _Scaler(np.asarray(mean), np.asarray(scale))


# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES
# ──────────────────────────────────────────────────────────────────────────────

def load_task1_model(pt_path: str):
    """
    Charge le MLP (task1, flat, distogram).
    Retourne (model_eval, scaler_or_None, feat_dim).
    """
    payload  = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd       = payload["state_dict"]
    arch     = _infer_mlp_arch(sd)
    feat_dim = payload.get("feat_dim") or arch.get("feat_dim")
    if feat_dim is None:
        raise ValueError(f"[Task1] Impossible d'inférer feat_dim depuis {pt_path}")

    model = MLPNet(
        d       = feat_dim,
        hidden1 = arch.get("hidden1", 256),
        hidden2 = arch.get("hidden2", 128),
        dropout = payload.get("best_params", {}).get("dropout", 0.3),
    )
    model.load_state_dict(sd)
    model.eval().to(DEVICE)

    scaler = _load_scaler(payload)
    print(f"  [Task1 MLP]  feat_dim={feat_dim}  "
          f"hidden1={arch.get('hidden1', 256)}  hidden2={arch.get('hidden2', 128)}  "
          f"scaler={'oui' if scaler else 'non'}")
    return model, scaler, feat_dim


def load_task2_model(pt_path: str, window_override: int = 0):
    """
    Charge le GRU (task2, window, distogram).
    Retourne (model_eval, scaler_or_None, feat_dim, window_used).
    """
    payload  = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd       = payload["state_dict"]
    arch     = _infer_gru_arch(sd)
    feat_dim = payload.get("feat_dim") or arch.get("feat_dim")
    if feat_dim is None:
        raise ValueError(f"[Task2] Impossible d'inférer feat_dim depuis {pt_path}")

    # window : CLI > valeur sauvegardée dans le .pt > défaut
    saved_win = None
    ws = payload.get("window_shape")        # (W, D) si présent
    if ws is not None:
        saved_win = int(ws[0])
    window = window_override or saved_win or WINDOW_SIZE

    model = GRUNet(
        d          = feat_dim,
        hidden     = arch.get("hidden", 128),
        num_layers = arch.get("num_layers", 2),
        dropout    = payload.get("best_params", {}).get("dropout", 0.3),
    )
    model.load_state_dict(sd)
    model.eval().to(DEVICE)

    scaler = _load_scaler(payload)
    print(f"  [Task2 GRU]  feat_dim={feat_dim}  "
          f"hidden={arch.get('hidden', 128)}  layers={arch.get('num_layers', 2)}  "
          f"window={window}  scaler={'oui' if scaler else 'non'}")
    return model, scaler, feat_dim, window


# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES EMBEDDINGS DISTOGRAM (sans labels)
# ──────────────────────────────────────────────────────────────────────────────

def _entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    (N_runs, L, L, 64) → (L, N_runs)
    Entropie de Shannon par run, moyennée sur l'axe distances.
    """
    logits = logits - logits.max(axis=-1, keepdims=True)   # stabilité numérique
    exp_l  = np.exp(logits)
    probs  = exp_l / exp_l.sum(axis=-1, keepdims=True)
    ent    = -(probs * np.log(probs + 1e-9)).sum(axis=-1)  # (N_runs, L, L)
    return ent.mean(axis=2).T                               # (L, N_runs)


def load_distogram_embeddings(h5_path: str, tag: str = "") -> dict:
    """
    Lit /proteins/<uid>/distogram_logits → entropie (L, DISTOGRAM_N_RUNS).
    Retourne {uid: np.ndarray(L, 25)}.
    Aucune clé 'labels' requise.
    """
    emb_dict = {}
    with h5py.File(h5_path, "r") as f:
        grp = f.get("proteins", f)
        for uid in grp.keys():
            g = grp[uid]
            if not hasattr(g, "keys") or "distogram_logits" not in g:
                print(f"  [SKIP {tag}] {uid} : clé 'distogram_logits' absente")
                continue
            try:
                L_attr = int(g.attrs["L"]) if "L" in g.attrs else None
            except Exception:
                L_attr = None

            logits = g["distogram_logits"][()].astype(np.float32)  # (N, L, L, 64)
            ent    = _entropy_from_logits(logits)                   # (L, N_actual)
            L      = L_attr if L_attr is not None else ent.shape[0]
            out    = np.zeros((L, DISTOGRAM_N_RUNS), dtype=np.float32)
            n_copy = min(ent.shape[1], DISTOGRAM_N_RUNS)
            out[:, :n_copy] = ent[:L, :n_copy]
            emb_dict[uid]   = out

    print(f"  [{tag}] {len(emb_dict)} protéines  ({Path(h5_path).name})")
    return emb_dict

# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES SÉQUENCES (CSV produit par fasta_to_input.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_sequences(csv_path: str) -> dict[str, str]:
    """
    Charge le CSV produit par fasta_to_input.py.
    Colonnes attendues : protein_id, sequence
    Retourne : {protein_id: sequence}
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "protein_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError(
            f"Le CSV {csv_path} doit contenir les colonnes 'protein_id' et 'sequence'. "
            f"Colonnes trouvées : {list(df.columns)}"
        )
    return dict(zip(df["protein_id"].astype(str), df["sequence"].astype(str)))

# ──────────────────────────────────────────────────────────────────────────────
# PRÉDICTION TASK 1  (MLP, flat)
# ──────────────────────────────────────────────────────────────────────────────

def predict_task1(model: nn.Module, scaler, embedding: np.ndarray) -> np.ndarray:
    """
    embedding : (L, D)   — embedding distogram task1 brut
    scaler    : _Scaler ou None, fitté sur (N_residus, D) en training

    Retourne score_disorder : (L,) ∈ [0, 1]
    """
    X = embedding.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)                 # (L, D) → (L, D)

    scores = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.FloatTensor(X)),
                                 batch_size=BATCH_SIZE):
            scores.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.concatenate(scores)               # (L,)


# ──────────────────────────────────────────────────────────────────────────────
# PRÉDICTION TASK 2  (GRU, window)
# ──────────────────────────────────────────────────────────────────────────────

def _build_windows(embedding: np.ndarray, window: int) -> np.ndarray:
    """
    Construit les fenêtres glissantes centrées pour tous les L résidus.
    embedding : (L, D)
    Retourne  : (L, window, D)
    """
    half   = window // 2
    padded = np.pad(embedding, ((half, half), (0, 0)), mode="constant")
    return np.stack(
        [padded[i: i + window] for i in range(embedding.shape[0])], axis=0
    ).astype(np.float32)                        # (L, W, D)


def predict_task2(model: nn.Module, scaler, embedding: np.ndarray,
                  window: int) -> np.ndarray:
    """
    embedding : (L, D)   — embedding distogram task2 brut (fichier H5 différent)
    scaler    : _Scaler ou None, fitté sur (N_windows, W*D) en training

    Retourne score_binding : (L,) ∈ [0, 1]  (calculé pour tous les résidus ;
    le masquage disorder-only se fait dans predict_hierarchical)
    """
    L, D    = embedding.shape
    windows = _build_windows(embedding, window)     # (L, W, D)

    if scaler is not None:
        # Le scaler training a été fitté sur le reshape (N, W*D)
        windows = scaler.transform(
            windows.reshape(L, -1)
        ).reshape(L, window, D)

    scores = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.FloatTensor(windows)),
                                 batch_size=BATCH_SIZE):
            scores.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.concatenate(scores)               # (L,)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE HIÉRARCHIQUE
# ──────────────────────────────────────────────────────────────────────────────

def predict_hierarchical(
    emb_t1: np.ndarray,            # embedding task1 (L, D1)  — H5 task1
    emb_t2: np.ndarray,            # embedding task2 (L, D2)  — H5 task2
    model_t1: nn.Module, scaler_t1,
    model_t2: nn.Module, scaler_t2,
    window: int,
    thr_t1: float = 0.5,
    thr_t2: float = 0.5,
) -> dict:
    """
    Pipeline hiérarchique pour une protéine.

    Les deux embeddings doivent avoir la même longueur L.
    Si les longueurs diffèrent (runs AF2 différents), on tronque au min(L1, L2)
    avec un avertissement.

    Retourne
    --------
    dict :
      "score_disorder"  : (L,)   P(disorder)         — issu de task1
      "score_binding"   : (L,)   P(binding|disorder) — issu de task2, 0 si structuré
      "predicted_class" : (L,)   int  0/1/2
    """
    L1, L2 = emb_t1.shape[0], emb_t2.shape[0]
    if L1 != L2:
        L = min(L1, L2)
        print(f"    [WARN] longueurs différentes t1={L1} t2={L2} → tronqué à {L}")
        emb_t1 = emb_t1[:L]
        emb_t2 = emb_t2[:L]
    else:
        L = L1

    # ── Task 1 : structure vs disorder (embedding H5 task1) ──────────────────
    score_disorder = predict_task1(model_t1, scaler_t1, emb_t1)    # (L,)

    # ── Task 2 : disorder vs binding (embedding H5 task2) ────────────────────
    # Calculé sur toute la séquence en une passe GPU, masqué ensuite
    score_binding_all = predict_task2(model_t2, scaler_t2, emb_t2, window)  # (L,)

    # ── Décision hiérarchique ─────────────────────────────────────────────────
    predicted_class = np.zeros(L, dtype=np.int32)
    score_binding   = np.zeros(L, dtype=np.float32)

    disordered_mask              = score_disorder >= thr_t1
    predicted_class[disordered_mask] = 1                            # disorder

    binding_mask                 = disordered_mask & (score_binding_all >= thr_t2)
    predicted_class[binding_mask]    = 2                            # disorder-binding
    score_binding[disordered_mask]   = score_binding_all[disordered_mask]

    return {
        "score_disorder":  score_disorder,
        "score_binding":   score_binding,
        "predicted_class": predicted_class,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ÉCRITURE CAID
# ──────────────────────────────────────────────────────────────────────────────

def write_caid(uid: str, result: dict, out_dir: Path,
                sequence: str | None = None,
                thr_t1: float = 0.5, thr_t2: float = 0.5) -> None:
    """
    Écrit deux fichiers au format CAID pour une protéine (comme predict_idr.py) :

      <out_dir>/disorder/<uid>.caid  → score_disorder, label = 1 si >= thr_t1
      <out_dir>/binding/<uid>.caid   → score_binding,  label = 1 si >= thr_t2

    Format :
      >uid
      1\tM\t0.8923\t1
      2\tE\t0.3210\t0
      ...
    """
    L = len(result["predicted_class"])
    seq = sequence if sequence is not None else "X" * L  # pas de séquence dispo ici

    disorder_dir = out_dir / "disorder"
    binding_dir  = out_dir / "binding"
    disorder_dir.mkdir(parents=True, exist_ok=True)
    binding_dir.mkdir(parents=True, exist_ok=True)

    # ── disorder ──
    with open(disorder_dir / f"{uid}.caid", "w") as fh:
        fh.write(f">{uid}\n")
        for i in range(L):
            score = result["score_disorder"][i]
            label = 1 if score >= thr_t1 else 0
            fh.write(f"{i + 1}\t{seq[i]}\t{score:.4f}\t{label}\n")

    # ── binding ──
    with open(binding_dir / f"{uid}.caid", "w") as fh:
        fh.write(f">{uid}\n")
        for i in range(L):
            score = result["score_binding"][i]
            label = 1 if score >= thr_t2 else 0
            fh.write(f"{i + 1}\t{seq[i]}\t{score:.4f}\t{label}\n")


# ──────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────────────────────
def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDEVICE : {DEVICE}")
    print("=" * 65)

    # ── Chargement des modèles ───────────────────────────────────────────────
    print("\nChargement des modèles")
    model_t1, scaler_t1, feat_dim_t1 = load_task1_model(args.model_task1)
    model_t2, scaler_t2, feat_dim_t2, window = load_task2_model(
        args.model_task2, args.window)

    # ── Chargement des séquences depuis le CSV ────────────────────────────────
    print(f"\nChargement des séquences : {args.csv}")
    sequences = load_csv_sequences(args.csv)
    print(f"  {len(sequences)} séquence(s) dans le CSV")

    # ── Chargement des embeddings (deux H5 distincts) ────────────────────────
    print("\nChargement des embeddings distogram")
    print(f"  Task1 H5 : {args.h5_task1}")
    emb_t1_dict = load_distogram_embeddings(args.h5_task1, tag="Task1")

    print(f"  Task2 H5 : {args.h5_task2}")
    emb_t2_dict = load_distogram_embeddings(args.h5_task2, tag="Task2")

    # Intersection des UIDs disponibles dans les deux H5 ET dans le CSV
    common_uids  = sorted(set(emb_t1_dict) & set(emb_t2_dict) & set(sequences))
    only_t1      = set(emb_t1_dict) - set(emb_t2_dict)
    only_t2      = set(emb_t2_dict) - set(emb_t1_dict)
    no_seq       = (set(emb_t1_dict) & set(emb_t2_dict)) - set(sequences)

    if only_t1:
        print(f"  [WARN] {len(only_t1)} protéine(s) présentes dans h5_task1 "
              f"uniquement → ignorées : {sorted(only_t1)[:5]}{'...' if len(only_t1)>5 else ''}")
    if only_t2:
        print(f"  [WARN] {len(only_t2)} protéine(s) présentes dans h5_task2 "
              f"uniquement → ignorées : {sorted(only_t2)[:5]}{'...' if len(only_t2)>5 else ''}")
    if no_seq:
        print(f"  [WARN] {len(no_seq)} protéine(s) sans séquence dans le CSV "
              f"→ ignorées : {sorted(no_seq)[:5]}{'...' if len(no_seq)>5 else ''}")
    if not common_uids:
        raise RuntimeError("Aucun UID commun entre h5_task1, h5_task2 et le CSV — "
                           "vérifier les fichiers d'entrée.")
    print(f"  → {len(common_uids)} protéines communes à traiter")

    # ── Prédiction + écriture CAID ───────────────────────────────────────────
    print(f"\nPrédiction  "
          f"(thr_task1={args.threshold_task1}  thr_task2={args.threshold_task2}  "
          f"window={window})")
    print("=" * 65)

    n_struct = n_disorder = n_binding = 0

    for uid in common_uids:
        emb_t1 = emb_t1_dict[uid]
        emb_t2 = emb_t2_dict[uid]

        result = predict_hierarchical(
            emb_t1    = emb_t1,
            emb_t2    = emb_t2,
            model_t1  = model_t1,
            scaler_t1 = scaler_t1,
            model_t2  = model_t2,
            scaler_t2 = scaler_t2,
            window    = window,
            thr_t1    = args.threshold_task1,
            thr_t2    = args.threshold_task2,
        )

        L   = len(result["predicted_class"])
        seq = sequences[uid][:L]   # tronquée à la même longueur que les embeddings
        if len(seq) < L:
            print(f"    [WARN] {uid} : séquence CSV plus courte ({len(seq)}) "
                  f"que les embeddings ({L}) → tronqué à {len(seq)}")
            L = len(seq)
            result = {k: v[:L] for k, v in result.items()}

        write_caid(uid, result, out_dir, sequence=seq,
                   thr_t1=args.threshold_task1, thr_t2=args.threshold_task2)

        counts      = np.bincount(result["predicted_class"], minlength=3)
        n_struct   += counts[0]
        n_disorder += counts[1]
        n_binding  += counts[2]

        print(f"  {uid:30s}  L={L:5d}  "
              f"struct={counts[0]:5d}  "
              f"disorder={counts[1]:5d}  "
              f"binding={counts[2]:5d}")

    total = n_struct + n_disorder + n_binding
    print("=" * 65)
    print(f"\nRÉSUMÉ — {len(common_uids)} protéines  |  {total:,} résidus")
    print(f"  Structure        : {n_struct:>8,}  ({100*n_struct/max(total,1):.1f} %)")
    print(f"  Disorder         : {n_disorder:>8,}  ({100*n_disorder/max(total,1):.1f} %)")
    print(f"  Disorder-binding : {n_binding:>8,}  ({100*n_binding/max(total,1):.1f} %)")
    print(f"\n  Fichiers CAID → {out_dir.resolve()}/")
    print("=" * 65)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prédiction hiérarchique IDP (distogram, sans labels) → CAID"
    )
    p.add_argument("--model_task1",     required=True,
                   help="Chemin vers le .pt du MLP task1 (structure vs disorder)")
    p.add_argument("--model_task2",     required=True,
                   help="Chemin vers le .pt du GRU task2 (disorder vs binding)")
    p.add_argument("--csv",             required=True,
                   help="CSV produit par fasta_to_input.py (colonnes : protein_id, sequence)")
    p.add_argument("--h5_task1",        required=True,
                   help="H5 distogram utilisé pour task1 (même que son training)")
    p.add_argument("--h5_task2",        required=True,
                   help="H5 distogram utilisé pour task2 (même que son training)")
    p.add_argument("--out_dir",         default="caid_predictions",
                   help="Répertoire de sortie (défaut : caid_predictions/)")
    p.add_argument("--threshold_task1", type=float, default=0.5,
                   help="Seuil score_disorder pour classer 'désordonné' (défaut 0.5)")
    p.add_argument("--threshold_task2", type=float, default=0.5,
                   help="Seuil score_binding pour classer 'binding' (défaut 0.5)")
    p.add_argument("--window",          type=int,   default=0,
                   help=f"Taille fenêtre GRU (0 = lire depuis le .pt, "
                        f"sinon écrase la valeur sauvegardée)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
