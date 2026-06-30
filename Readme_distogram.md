# Singularity image distogram.sif : Residue-level IDR Prediction Pipeline

Singularity image for residue-level IDP/IDR prediction into three classes:
**Structure**, **Disorder**, **Disorder-Binding**.

---
## Requirements

- Singularity ≥ 3.8
- Pre-computed AlphaFold2 outputs (distogram `.pickle` files per protein) --> can be run from our script `run_alphafold_cluster.sbatch`
- Input CSV file with columns `protein_id` and `sequence` --> directly output by `run_alphafold_cluster.sbatch`


## Usage

### Basic command

```bash
singularity run --nv --no-home \
    --bind <embeddings_dir>:<embeddings_dir> \
    --bind <output_dir>:<output_dir> \
    perceptron_complete.sif \
    prediction \
    --embeddings <embeddings_dir> \
    --csv        <embeddings_dir>/proteins.csv \
    --output     <output_dir>
```

### Available options

| Option | Required | Description |
|--------|----------|-------------|
| `--embeddings` | yes | Directory containing AF2 outputs per protein |
| `--csv` | yes | CSV file with columns `protein_id` and `sequence` |
| `--output` | yes | Output directory for `.caid` prediction files |
| `--embeddings_output` | no | Alternative directory for intermediate embeddings (default: `/opt/output`) |


### Expected AlphaFold2 output structure

```
<embeddings_dir>/
└── <protein_id>/
    ├── no_msa_recycle_0/
    │   └── *.pickle          # AF2 step 1 output
    └── no_msa_recycle_3/
        └── *.pickle          # AF2 step 2 output
```

### Input CSV format

```csv
protein_id,sequence
P04637,MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDP...
Q9Y4L1,MSSQPILENVSFKTLNDSGIELIGKSNVSRLQKLVTQDFNEQMRELLKAQLMQQ...
```

=> Both can be created by our script `run_alphafold_cluster.sbatch` from a FASTA files with multiples porteins as input

---

## Outputs

```
<output_dir>/
├── disorder/
│   └── <protein_id>.caid     # Task 1: Structure vs Disorder+Binding
├── binding/
│   └── <protein_id>.caid     # Task 2: Disorder vs Disorder-Binding
└── timings.csv               # Per-protein inference time
```

### CAID format

```
>P04637
1    M    0.8923    1
2    E    0.3210    0
3    E    0.7841    1
...
```

Columns: `position`, `amino_acid`, `score [0-1]`, `binary_label`.

- **disorder/*.caid** — probability of being disordered or disorder-binding (Task 1)
- **binding/*.caid** — probability of being a disorder-binding residue (Task 2)

---


## Image Contents

| Component | Details |
|---|---|
| Base image | `nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04` |
| Env `venv_idp` | Python 3.11 — distograms, prediction |
| Scripts | `/opt/scripts/` |
| ML models | `/opt/models/` |

### Internal scripts

```
/opt/scripts/
├── main_scripts.sh                  # Main orchestrator
├── prep_data_set.py                 # Distogram CSV preparation
├── create_input_data_distogram.py   # AF2 logits extraction → HDF5
└── run_prediction.py                # Hierarchical inference Task1 + Task2
```
---

## References

- AlphaFold2 distogram embeddings: AF2 internal output (`distogram/logits`, 64 bins)
- CAID format: [caid.idpcentral.org](https://caid.idpcentral.org)

---

*SPPICES Project — ANR-24-CE45-0866 — AMIG team, I2BC, CNRS*
*Contact: julie.daniel@i2bc.paris-saclay.fr*
*Contact: diego.zea@i2bc.paris-saclay.fr*