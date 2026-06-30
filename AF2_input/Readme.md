# First step to prepare the Input 

## ColabFold Batch Prediction Pipeline

This script takes as input a multi-sequence FASTA and runs [ColabFold](https://github.com/sokrypton/ColabFold) (AlphaFold2) structure predictions using an Apptainer/Singularity container on a SLURM cluster. It generates the output directory structure expected by downstream prediction analysis scripts.

---

### Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Pipeline Details](#pipeline-details)
- [Notes](#notes)

---

## Overview

The pipeline takes a multi-sequence FASTA file as input, parses it into individual protein entries, runs ColabFold on each protein with two configurations (no MSA, 0 or 3 recycling steps), and organizes the outputs into a structured directory tree. This output tree is the **required input format** for the downstream prediction and analysis scripts.

---

## Requirements

### Python

Only one Python dependency is required:

```bash
pip install pandas
```

The `fasta_to_input.py` helper script uses only `pandas` (plus Python standard library modules).

### ColabFold Singularity/Apptainer Image

The pipeline uses the **ColabFold 1.5.5** container image. You can download it from Zenodo:

> **ColabFold 1.5.5 Apptainer/Singularity Image**
> [https://zenodo.org/records/20842530](https://zenodo.org/records/20842530)

Download the `.sif` file and place it in a location accessible from your compute nodes. You will also need a local ColabFold **cache directory** (databases, parameter weights). Refer to the [ColabFold documentation](https://github.com/sokrypton/ColabFold) for cache setup instructions.

### SLURM Cluster

The script is designed to run as a SLURM job. Required resources per job (adjust as needed):

| Resource       | Default value |
|----------------|---------------|
| Memory         | 30 GB         |
| GPU            | 1             |
| CPUs per task  | 8             |
| Wall time      | 20 hours      |

---

## Installation

1. Copy the 2 requiered scripts into your working directory:

```
pipeline_input_preparation/
├── run_alphafold_cluster.sbatch   # Main SLURM script (this file)
└── fasta_to_input.py           # FASTA parsing helper
```

2. Edit the following paths inside `run_alphafold_cluster.sbatch ` to match your environment:

```bash
SIF_PATH=/path/to/sif_images/ColabFold_AF2_1-5-5/colabfold-1.5.5-cuda12.2.2.sif
CACHE=/path/to/sif_images/ColabFold_AF2_1-5-5/cache
```

3. Ensure `fasta_to_input.py` is in the same directory as the SLURM script (it is located via `$SLURM_SUBMIT_DIR`).

---

## Usage

Submit the job with:

```bash
sbatch run_alphafold_cluster.sbatch <FASTA_FILE> <OUTPUT_DIR>
```

**Arguments:**

| Argument       | Description                                                      |
|----------------|------------------------------------------------------------------|
| `FASTA_FILE`   | Path to a multi-sequence FASTA file containing all target proteins |
| `OUTPUT_DIR`   | Path to the root output directory (will be created if needed)   |


---

## Output Structure

For each protein in the FASTA file, two ColabFold runs are performed (0 and 3 recycling steps, both in `single_sequence` MSA mode). The resulting output directory is organized as follows:

```
OUTPUT_DIR/
└── <protein_id>/
    ├── sequences.a3m               # Input MSA file for ColabFold
    ├── no_msa_recycle_0/           # Run with 0 recycling steps
    │   └── predictions/
    │       ├── config.json
    │       ├── log.txt
    │       ├── cite.bibtex
    │       └── sequences/
    │           ├── sequences.a3m
    │           ├── sequences.done.txt
    │           ├── models/         # Predicted .pdb structures (5 models × 5 seeds)
    │           ├── plots/          # PAE / pLDDT plots (.png)
    │           └── scores/         # Per-model score files (.json)
    └── no_msa_recycle_3/           # Run with 3 recycling steps
        └── predictions/
            └── sequences/
                ├── models/
                ├── plots/
                └── scores/
```

> **Important:** This `OUTPUT_DIR` structure is the **required input** for the downstream prediction and analysis scripts. Do not manually reorganize it.

---

## Pipeline Details

### FASTA Parsing (`fasta_to_input.py`)

- Reads the input multi-sequence FASTA file.
- Creates one subdirectory per protein under `OUTPUT_DIR`.
- Generates a `proteins.csv` file listing all `protein_id` values (used to drive the loop).

### ColabFold Runs

For each protein, the pipeline runs ColabFold twice:

| Run                  | MSA mode          | Recycling steps |
|----------------------|-------------------|-----------------|
| `no_msa_recycle_0`   | `single_sequence` | 0               |
| `no_msa_recycle_3`   | `single_sequence` | 3               |

Each run generates **5 models × 5 random seeds** with dropout enabled, using a randomly sampled seed (range 10000–99999) logged at job start for reproducibility.

Runs are **skipped** if the output directory already exists, allowing safe resubmission of partially failed jobs.

### File Organization

After each ColabFold run, output files are automatically reorganized into the structured `predictions/sequences/{models,plots,scores}/` hierarchy described above.

---

## Notes

- **Reproducibility:** The random seed is printed at the start of the job output log (`*.out` file). Save it if you need to reproduce a specific run.
- **Skipping completed runs:** The pipeline checks for existing output directories before running. If a protein has already been processed, it is skipped automatically.
- **GPU requirement:** The Singularity container requires GPU access (`--nv` flag). Ensure your SLURM partition has GPU nodes available.
- **Temporary files:** `$TMPDIR` and `$TMP` are set to `$JOBSCRATCH` to avoid filling shared `/tmp` space on compute nodes.
