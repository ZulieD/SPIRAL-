set -e
set -x

# ─────────────────────────── ARGUMENT UTILISATEUR ────────────────────────────
if [ -z "$1" ]; then
    echo "[ERROR] Usage: $0 <EMBEDDINGS_DIR_TEST>"
    exit 1
fi
EMBEDDINGS_DIR_TEST="$1"

if [ -z "$2" ]; then
    echo "[ERROR] Usage: $0 <CSV_INPUT>"
    exit 1
fi
CSV_INPUT="$2"

if [ -z "$3" ]; then
    echo "[ERROR] Usage: $0 <FINAL_OUTPUT_FOLDER>"
    exit 1
fi
FINAL_OUTPUT_FOLDER="$3"

# ─────────────────────────── GLOBAL CONFIG ───────────────────────────────────
PYTHON_SCRIPTS_DIR="/opt/scripts"
if [ -n "$4" ]; then
    # Chemin fourni par l'utilisateur via --embeddings_output
    BASE_OUTPUT_DIR="$4"
    mkdir -p "$BASE_OUTPUT_DIR"
else
    # Tmpdir automatique, supprimé à la fin
    BASE_OUTPUT_DIR=$(mktemp -d /tmp/triad_embeddings_XXXXXX)
    trap "rm -rf '$BASE_OUTPUT_DIR'" EXIT
    echo "[INFO] Embeddings intermédiaires dans : $BASE_OUTPUT_DIR (supprimé à la fin)"
fi

STEP1_FOLDER="no_msa_recycle_0"
STEP2_FOLDER="no_msa_recycle_3"

OUTPUT_DIR_DIST="${BASE_OUTPUT_DIR}/two_h5_s1_${STEP1_FOLDER}_s2_${STEP2_FOLDER}_complete"
mkdir -p "$OUTPUT_DIR_DIST"

# ─────────────────────────── GPU CHECK ───────────────────────────────────────
echo "=============================="
echo "Job started  : $(date)"
echo "Node         : $(hostname)"
echo "GPU(s)       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "CUDA visible : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=============================="

# ─────────────────────────── DÉTECTION DES PROTÉINES ─────────────────────────
# Récupère tous les dossiers protéines présents dans EMBEDDINGS_DIR_TEST/STEP_FOLDER
detect_proteins_from_dir() {
    local base_dir="$1"
    local step_folder="$2"
    find "$base_dir" -mindepth 1 -maxdepth 1 -type d \
        |while read -r prot_dir; do
            if [ -d "$prot_dir/$step_folder" ]; then
                basename "$prot_dir"
            fi
        done
}

mapfile -t P_S1_TEST < <(detect_proteins_from_dir "$EMBEDDINGS_DIR_TEST" "$STEP1_FOLDER")
mapfile -t P_S2_TEST < <(detect_proteins_from_dir "$EMBEDDINGS_DIR_TEST" "$STEP2_FOLDER")

# ─────────────────────────── VALIDATION ──────────────────────────────────────
for count in "${#P_S1_TEST[@]}" "${#P_S2_TEST[@]}"; do
    [ "$count" -eq 0 ] && echo "[ERROR] Empty protein list" && exit 1
done

echo "P_S1_TEST: ${#P_S1_TEST[@]} proteins"
echo "P_S2_TEST: ${#P_S2_TEST[@]} proteins"

CSV_S1_TEST=$(IFS=','; echo "${P_S1_TEST[*]}")
CSV_S2_TEST=$(IFS=','; echo "${P_S2_TEST[*]}")

# ─────────────────────────── STEP 1 TEST ─────────────────────────────────────
if [ ! -f "$OUTPUT_DIR_DIST/s1_prepared_embeddings_test.h5" ]; then
    python3 "$PYTHON_SCRIPTS_DIR/prep_data_set.py" \
        --protein_id   "$CSV_S1_TEST" \
        --folder_paths "$EMBEDDINGS_DIR_TEST" \
        --output_file  "$OUTPUT_DIR_DIST/s1_prep_test.csv" \
        --msa_folder   "$STEP1_FOLDER"
    python3 "$PYTHON_SCRIPTS_DIR/create_input_data_distogram.py" \
        --input_csv  "$OUTPUT_DIR_DIST/s1_prep_test.csv" \
        --output_dir "$OUTPUT_DIR_DIST/tmp_s1_test" \
        --n_workers  "${SLURM_CPUS_PER_TASK:-4}"
    mv "$OUTPUT_DIR_DIST/tmp_s1_test/prepared_distograms.h5" \
       "$OUTPUT_DIR_DIST/s1_prepared_embeddings_test.h5"
    rm -rf "$OUTPUT_DIR_DIST/tmp_s1_test"
else
    echo "  s1_prepared_embeddings_test.h5 exists — skipping"
fi

# ─────────────────────────── STEP 2 TEST ─────────────────────────────────────
if [ ! -f "$OUTPUT_DIR_DIST/s2_prepared_embeddings_test.h5" ]; then   
    python3 "$PYTHON_SCRIPTS_DIR/prep_data_set.py" \
        --protein_id   "$CSV_S2_TEST" \
        --folder_paths "$EMBEDDINGS_DIR_TEST" \
        --output_file  "$OUTPUT_DIR_DIST/s2_prep_test.csv" \
        --msa_folder   "$STEP2_FOLDER"                            
    python3 "$PYTHON_SCRIPTS_DIR/create_input_data_distogram.py" \
        --input_csv  "$OUTPUT_DIR_DIST/s2_prep_test.csv" \
        --output_dir "$OUTPUT_DIR_DIST/tmp_s2_test" \
        --n_workers  "${SLURM_CPUS_PER_TASK:-4}"
    mv "$OUTPUT_DIR_DIST/tmp_s2_test/prepared_distograms.h5" \
       "$OUTPUT_DIR_DIST/s2_prepared_embeddings_test.h5"               
    rm -rf "$OUTPUT_DIR_DIST/tmp_s2_test"
else
    echo "  s2_prepared_embeddings_test.h5 exists — skipping"
fi
echo "=============================="
echo "Embeddings finished : $(date)"
echo "=============================="


python "$PYTHON_SCRIPTS_DIR/run_predictions.py" \
    --model_task1 /opt/models/full_task1_struct_vs_disorder_meanJ_MLP_optimized_MLP.pt \
    --model_task2 /opt/models/full_task2_disorder_vs_binding_conv2d_GRU_optimized_GRU.pt \
    --h5_task1        "$OUTPUT_DIR_DIST/s1_prepared_embeddings_test.h5" \
    --h5_task2        "$OUTPUT_DIR_DIST/s2_prepared_embeddings_test.h5" \
    --out_dir         "$FINAL_OUTPUT_FOLDER" \
    --threshold_task1 0.5 \
    --threshold_task2 0.5

echo "=============================="
echo "Job finished : $(date)"
echo "=============================="