#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import os
import glob

############ INITIALIZATION ############
parser = argparse.ArgumentParser()
parser.add_argument('--protein_id', type=str, required=True)
parser.add_argument('--folder_paths', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--msa_folder', type=str, default="msa_recycle_3")
args = parser.parse_args()

rows = []

for protein_id in args.protein_id.split(','):
    print(f"\n[INFO] Traitement de la protéine : {protein_id}")

    folder_prot = os.path.join(args.folder_paths, protein_id)

    pickle_files = glob.glob(os.path.join(folder_prot, args.msa_folder, "*.pickle"))

    if not pickle_files:
        print(f"[WARNING] Aucun fichier pickle trouvé pour {protein_id}")
        continue

    rows.append({
        'protein_id': protein_id,
        'pickle_files': ';'.join(pickle_files),
    })

# Création dataframe final
df_new = pd.DataFrame(rows)
df_new.to_csv(args.output_file, index=False)

print(f"\n[INFO] Terminé. {len(rows)} protéines écrites dans {args.output_file}")
