# Singularity image spiral.sif : Residue-level IDR Prediction Pipeline

Singularity image for residue-level IDP/IDR prediction into three classes:
**Structure**, **Disorder**, **Disorder-Binding**.

---
## Requirements

- Singularity ≥ 3.8
- Pre-computed AlphaFold2 outputs (distogram `.pickle` files per protein) --> can be run from our script `run_alphafold_cluster.sbatch`
- Input CSV file with columns `protein_id` and `sequence` --> directly output by `run_alphafold_cluster.sbatch`


## Usage

### SPIRAL Singularity/Apptainer Image

The pipeline uses a container image. You can download it from Zenodo:

> **SPIRAL Apptainer/Singularity Image**
> [(https://zenodo.org/records/20842530)](https://zenodo.org/records/21098730)

Download the `.sif` file and place it in a location accessible from your compute nodes. 

### Basic command

```bash
singularity run --nv --no-home \
    --bind <embeddings_dir>:<embeddings_dir> \
    --bind <output_dir>:<output_dir> \
    spiral.sif \
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
├── P04637.tsv
└── Q9Y4L1.tsv              # Per-protein inference time
```

### tsv with CAID format

```
# uid=P04637
residue_index	score_disorder	score_binding	predicted_class
    1	           0.682276	      0.491631	         1
   2	           0.672031	      0.475615	         1
   3	           0.672628	      0.453714	         1
...
```

- score_disorder — probability of being disordered or disorder-binding (Task 1)
- score_binding — probability of being a disorder-binding residue (Task 2)
- predicted_class - 0 to 2 (0=structure; 1=disorder; 2=disorder-binding)

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
