#!/usr/bin/env python3
"""
fasta_to_input.py
-----------------------
Parses a multi-sequence FASTA file and, for each protein:
- creates a directory <output_dir>/<protein_id>/
- writes a sequences.a3m file containing the sequence
- saves the information to a CSV with the columns:
protein_id, sequence

Usage:
python fasta_to_input.py \\
--fasta  proteins.fasta \\
--output_dir ./proteins \\
--output_csv proteins.csv
"""

import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Parser FASTA
# ---------------------------------------------------------------------------

def parse_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """
    Parse un fichier FASTA multi-séquences.
    Retourne une liste de tuples (protein_id, sequence).
    L'ID est le premier mot de la ligne d'en-tête (sans le '>').
    """
    entries = []
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    entries.append((current_id, "".join(current_seq)))
                current_id  = line[1:].split()[0]   # premier mot après '>'
                current_seq = []
            else:
                current_seq.append(line.upper())

    # Dernière séquence
    if current_id is not None and current_seq:
        entries.append((current_id, "".join(current_seq)))

    return entries


# ---------------------------------------------------------------------------
# Création des dossiers et fichiers .a3m
# ---------------------------------------------------------------------------

def write_a3m(protein_id: str, sequence: str, output_dir: str) -> None:
    """
    Crée <output_dir>/<protein_id>/sequences.a3m avec la séquence au format FASTA.
    """
    folder = os.path.join(output_dir, protein_id)
    os.makedirs(folder, exist_ok=True)

    a3m_path = os.path.join(folder, "sequences.a3m")
    with open(a3m_path, "w") as f:
        f.write(f">{protein_id}\n{sequence}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FASTA multi-séquences → dossiers par protéine + CSV"
    )
    parser.add_argument("--fasta",       required=True,
                        help="Fichier FASTA d'entrée (multi-séquences)")
    parser.add_argument("--output_dir",  required=True,
                        help="Dossier racine dans lequel créer les sous-dossiers par protéine")
    parser.add_argument("--output_csv",  required=True,
                        help="Chemin du CSV de sortie (protein_id, sequence)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.fasta):
        print(f"ERREUR : fichier introuvable → {args.fasta}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    entries = parse_fasta(args.fasta)
    if not entries:
        print("ERREUR : aucune séquence trouvée dans le fichier FASTA.")
        sys.exit(1)

    print(f"[INFO] {len(entries)} séquences trouvées dans {args.fasta}")

    rows = []
    for protein_id, sequence in entries:
        write_a3m(protein_id, sequence, args.output_dir)
        rows.append({"protein_id": protein_id, "sequence": sequence})
        print(f"  [OK] {protein_id}  ({len(sequence)} résidus) → {args.output_dir}/{protein_id}/sequences.a3m")

    df = pd.DataFrame(rows, columns=["protein_id", "sequence"])
    df.to_csv(args.output_csv, index=False)

    print(f"\n[INFO] CSV écrit → {args.output_csv}  ({len(rows)} entrées)")
    print("[INFO] Terminé.")


if __name__ == "__main__":
    main()