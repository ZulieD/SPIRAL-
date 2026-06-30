#!/usr/bin/env bash
#SBATCH --job-name=idp_multicouche
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=80:00:00
#SBATCH --output=run_test.o%j.out

cd "${SLURM_SUBMIT_DIR}"         
AF2_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/test"
mkdir -p "$AF2_OUTPUT_DIR"       

FASTA_FILE="test_fasta.fasta"

AF2_JOB_ID=$(sbatch -p amig --parsable \
    run_AF2/run_alphafold_cluster.sbatch "$FASTA_FILE" "$AF2_OUTPUT_DIR")
echo "AF2 job soumis : $AF2_JOB_ID"

FINAL_OUTPUT="${AF2_OUTPUT_DIR}/final_output"
mkdir -p "$FINAL_OUTPUT"          

#SIF_PATH="perceptron_complete.sif"
SIF_PATH="distogram.sif"

sbatch -p amig \
    --dependency=afterok:"$AF2_JOB_ID" \
    --job-name=idp_prediction \
    --cpus-per-task=4 \
    --mem=50G \
    --gres=gpu:1 \
    --time=10:00:00 \
    --wrap="singularity run --nv --no-home \
        --bind ${AF2_OUTPUT_DIR}:${AF2_OUTPUT_DIR} \
        --bind ${FINAL_OUTPUT}:${FINAL_OUTPUT} \
        ${SIF_PATH} \
        prediction \
        --embeddings ${AF2_OUTPUT_DIR} \
        --csv ${AF2_OUTPUT_DIR}/proteins.csv \
        --output ${FINAL_OUTPUT}"