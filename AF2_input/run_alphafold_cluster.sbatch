#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=run_AF2.jl.o%j.out

set -e
set -x
source /stockage/EQUIPES/AMIG/MEMBERS/julie.daniel/IDP/venv_idp/bin/activate
cd "${SLURM_SUBMIT_DIR}"

start=$(date +%s)

export TMP=$JOBSCRATCH
export TMPDIR=$JOBSCRATCH

SIF_PATH=/path/to/sif_images/ColabFold_AF2_1-5-5/colabfold-1.5.5-cuda12.2.2.sif
CACHE=/path/to/sif_images/ColabFold_AF2_1-5-5/cache

SIF_PATH=/stockage/EQUIPES/AMIG/SCRIPTS/sif_images/ColabFold_AF2_1-6-1/colabfold_1.6.1.sif
CACHE=/stockage/EQUIPES/AMIG/SCRIPTS/sif_images/ColabFold_AF2_1-6-1/cache

# ── Arguments ────────────────────────────────────────────────────────────────
# $1 : chemin vers le fichier FASTA multi-séquences
FASTA_FILE="$1"
MAIN_OUTPUT_DIR="$2"


PYTHON_SCRIPT="${SLURM_SUBMIT_DIR}/run_AF2/fasta_to_input.py"

SEED=$(( RANDOM % 90000 + 10000 ))
echo "Random seed: $SEED"

################################
# PARSE FASTA → dossiers + CSV
################################

OUTPUT_CSV="${MAIN_OUTPUT_DIR}/proteins.csv"

echo "Parsing FASTA : $FASTA_FILE"
python3 "$PYTHON_SCRIPT" \
    --fasta      "$FASTA_FILE" \
    --output_dir "$MAIN_OUTPUT_DIR" \
    --output_csv "$OUTPUT_CSV"

# Récupère la liste des protein_id depuis le CSV (skip header)
mapfile -t QUERYS < <(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f1)

echo "Protéines à traiter : ${QUERYS[*]}"

################################
# FUNCTION TO ORGANIZE FILES
################################
organize_files() {
    local OUTPUT_DIR="$1"

    echo "Reorganizing files in $OUTPUT_DIR ..."
    cd "$OUTPUT_DIR" || exit 1

    PREDICTIONS_DIR="predictions"
    mkdir -p "$PREDICTIONS_DIR"

    mv -f config.json   "$PREDICTIONS_DIR/" 2>/dev/null || true
    mv -f log.txt       "$PREDICTIONS_DIR/" 2>/dev/null || true
    mv -f cite.bibtex   "$PREDICTIONS_DIR/" 2>/dev/null || true

    SEQ_DIR="$PREDICTIONS_DIR/sequences"
    mkdir -p "$SEQ_DIR"

    mv -f sequences.a3m       "$SEQ_DIR/" 2>/dev/null || true
    mv -f sequences.done.txt  "$SEQ_DIR/" 2>/dev/null || true

    MODELS_DIR="$SEQ_DIR/models"
    mkdir -p "$MODELS_DIR"

    shopt -s nullglob
    for file in *.pdb; do
        mv -f "$file" "$MODELS_DIR/"
    done

    PLOTS_DIR="$SEQ_DIR/plots"
    mkdir -p "$PLOTS_DIR"
    for file in *.png; do
        mv -f "$file" "$PLOTS_DIR/"
    done

    SCORES_DIR="$SEQ_DIR/scores"
    mkdir -p "$SCORES_DIR"
    for file in *.json; do
        mv -f "$file" "$SCORES_DIR/"
    done
    shopt -u nullglob

    echo "Reorganization finished."
}

################################
# FUNCTION TO RUN COLABFOLD
################################
run_colabfold () {

    local INPUT_DIR="$1"
    local INPUT_FILE="$2"
    local OUTPUT_DIR="$3"
    local NUM_RECYCLE="$4"
    local MSA_MODE="$5"

    echo "Output directory: $OUTPUT_DIR"

    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "Already processed, skipping."
        return
    fi

    if [[ -d "$OUTPUT_DIR/predictions/sequences/models" ]]; then
        echo "Already processed, skipping."
        return
    fi

    if [[ ! -f "$INPUT_DIR/$INPUT_FILE" ]]; then
        echo "Input file $INPUT_DIR/$INPUT_FILE not found. Skipping."
        return
    fi

    echo "Running ColabFold for $INPUT_FILE with MSA mode '$MSA_MODE' and recycle count $NUM_RECYCLE..."
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    singularity exec --nv --no-home --cleanenv \
        --bind "$INPUT_DIR:/mnt/input" \
        --bind "$OUTPUT_DIR:/mnt/output" \
        --bind "$CACHE:/cache" \
        "$SIF_PATH" \
        colabfold_batch \
        "/mnt/input/$INPUT_FILE" /mnt/output \
        ${MSA_MODE:+--msa-mode "$MSA_MODE"} \
        --save-all \
        --random-seed "$SEED" \
        --num-seeds 5 \
        --use-dropout \
        --num-models 5 \
        --num-recycle "$NUM_RECYCLE" \
        --overwrite-existing-results

    organize_files "$OUTPUT_DIR"
    cd "${SLURM_SUBMIT_DIR}"
}


################################
# MAIN
################################

for QUERY in "${QUERYS[@]}"; do

    echo "Processing query: $QUERY"
    INPUT_DIR="$MAIN_OUTPUT_DIR/$QUERY"

    run_colabfold "$INPUT_DIR" "sequences.a3m" "$INPUT_DIR/no_msa_recycle_0" 0 single_sequence
    run_colabfold "$INPUT_DIR" "sequences.a3m" "$INPUT_DIR/no_msa_recycle_3" 3 single_sequence

done

echo "All ColabFold runs are finished."

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"