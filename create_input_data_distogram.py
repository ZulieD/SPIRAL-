
#!/usr/bin/env python3
"""
Prépare les embeddings distogrammes AlphaFold2 en parallèle.
 
Structure HDF5 final :
  /proteins/<protein_id>/
      distogram_logits    (n_models, L, L, B)
      distogram_bin_edges (n_models, B_edges)
      attrs["n_models"]   int
      attrs["L"]          int
  /protein_ids            liste des IDs (bytes)
 
Usage:
  python create_input_data_distogram.py \\
    --input_csv  dataset.csv \\
    --output_dir ./embeddings_distogram \\
    --n_workers  16 \\
    --timeout    300
"""
 
import argparse
import multiprocessing as mp
import os
import pickle
import shutil
import time
 
import h5py
import numpy as np
import pandas as pd
 
 
# ─────────────────────────── pre-flight checks ───────────────────────────────
 
def check_pickle_file(path):
    path = path.strip()
    if not path:
        return False, "empty path"
    if not os.path.exists(path):
        return False, f"file not found: {path}"
    if os.path.getsize(path) == 0:
        return False, f"empty file (0 bytes): {path}"
    try:
        with open(path, "rb") as f:
            header = f.read(2)
        if len(header) < 2 or header[0] != 0x80:
            return False, f"invalid pickle header: {path}"
    except OSError as e:
        return False, f"cannot read: {path} — {e}"
    return True, "ok"
 
 
def check_task(task):
    warnings = []
    valid_files = []
    for pf in task["pickle_files"]:
        ok, reason = check_pickle_file(pf)
        if ok:
            valid_files.append(pf)
        else:
            warnings.append(f"  [PRE-CHECK SKIP] {task['pid']} — {reason}")
 
    if len(valid_files) == 0:
        return False, warnings + [f"  [PRE-CHECK FAIL] {task['pid']} — 0 valid pickle files"]
 
    if len(valid_files) < len(task["pickle_files"]):
        warnings.append(
            f"  [PRE-CHECK WARN] {task['pid']} — "
            f"{len(valid_files)}/{len(task['pickle_files'])} valid files")
 
    task["pickle_files"] = valid_files
    return True, warnings
 
 
# ─────────────────────────── worker ─────────────────────────────────────────
 
def _load_pickle_with_timeout(pf, timeout_sec=120):
    import threading
    import queue as queue_mod
 
    result_q = queue_mod.Queue()
 
    def _load():
        try:
            with open(pf, "rb") as f:
                data = pickle.load(f)
            result_q.put(("ok", data))
        except Exception as e:
            result_q.put(("fail", str(e)))
 
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
 
    if t.is_alive():
        return False, f"timeout after {timeout_sec}s (thread still alive)"
 
    try:
        status, payload = result_q.get_nowait()
        return (True, payload) if status == "ok" else (False, payload)
    except Exception:
        return False, "no result from thread"
 
 
def process_protein(task):
    pid          = task["pid"]
    pickle_files = task["pickle_files"]
    tmp_path     = task["tmp_path"]
    overwrite    = task["overwrite"]
    timeout_sec  = task.get("timeout", 300)
    per_file_timeout = min(timeout_sec, 120)
 
    if os.path.exists(tmp_path) and not overwrite:
        return (pid, "skip", "already exists")
 
    n_models = len(pickle_files)
 
    disto_logits_list    = []
    disto_bin_edges_list = []
    n_loaded = 0
    skipped_pickles = []
    L = None
 
    for pf in pickle_files:
        pf = pf.strip()
 
        ok, payload = _load_pickle_with_timeout(pf, per_file_timeout)
        if not ok:
            skipped_pickles.append(f"{payload}:{os.path.basename(pf)}")
            continue
        data = payload
 
        try:
            if not isinstance(data, dict):
                skipped_pickles.append(f"not_a_dict:{os.path.basename(pf)}")
                continue
            if "distogram" not in data:
                skipped_pickles.append(f"missing_key:distogram:{os.path.basename(pf)}")
                continue
            if "logits" not in data["distogram"]:
                skipped_pickles.append(f"missing_key:logits:{os.path.basename(pf)}")
                continue
            if "bin_edges" not in data["distogram"]:
                skipped_pickles.append(f"missing_key:bin_edges:{os.path.basename(pf)}")
                continue
 
            logits    = data["distogram"]["logits"]    # (L, L, B)
            bin_edges = data["distogram"]["bin_edges"] # (B_edges,)
 
            # Infer L from first valid pickle
            if L is None:
                L = logits.shape[0]
 
            if logits.ndim != 3 or logits.shape[0] != L or logits.shape[1] != L:
                skipped_pickles.append(f"bad_shape:logits{logits.shape}:{os.path.basename(pf)}")
                continue
            if bin_edges.ndim != 1:
                skipped_pickles.append(f"bad_shape:bin_edges{bin_edges.shape}:{os.path.basename(pf)}")
                continue
            if not np.isfinite(logits).all():
                skipped_pickles.append(f"nan_inf:logits:{os.path.basename(pf)}")
                continue
 
        except Exception as e:
            skipped_pickles.append(f"validate_error({e}):{os.path.basename(pf)}")
            continue
 
        disto_logits_list.append(logits.astype(np.float32))
        disto_bin_edges_list.append(bin_edges.astype(np.float32))
        n_loaded += 1
 
    if n_loaded == 0:
        reason = f"0/{n_models} pickles loaded"
        if skipped_pickles:
            reason += f" | skipped: {'; '.join(skipped_pickles[:5])}"
        return (pid, "fail", reason)
 
    try:
        disto_log_arr   = np.stack(disto_logits_list)    # (n_models, L, L, B)
        disto_edges_arr = np.stack(disto_bin_edges_list) # (n_models, B_edges)
 
        with h5py.File(tmp_path, "w") as hf:
            hf.create_dataset("distogram_logits",    data=disto_log_arr,
                              compression="gzip", compression_opts=4)
            hf.create_dataset("distogram_bin_edges", data=disto_edges_arr,
                              compression="gzip", compression_opts=4)
            hf.attrs["n_models"] = n_loaded
            hf.attrs["L"]        = L
 
        msg = (f"{n_loaded}/{n_models} models | "
               f"distogram_logits={disto_log_arr.shape}")
        if skipped_pickles:
            msg += f" | {len(skipped_pickles)} pickles skipped"
        return (pid, "ok", msg)
 
    except Exception as e:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass
        return (pid, "fail", f"H5 write error: {e}")
 
 
# ─────────────────────────── merge ──────────────────────────────────────────
 
def merge(tmp_dir, output_file, ok_pids):
    print(f"\n[MERGE] {len(ok_pids)} proteins → {output_file}")
    saved = []
    with h5py.File(output_file, "w") as hf_out:
        grp = hf_out.require_group("proteins")
        for pid in ok_pids:
            tmp_path = os.path.join(tmp_dir, f"{pid}.h5")
            if not os.path.exists(tmp_path):
                print(f"  [WARN] {pid}: tmp file missing, skipped")
                continue
            try:
                with h5py.File(tmp_path, "r") as hf_in:
                    prot_grp = grp.require_group(pid)
                    for key in ["distogram_logits", "distogram_bin_edges"]:
                        if key in hf_in:
                            hf_in.copy(key, prot_grp)
                    prot_grp.attrs["n_models"] = hf_in.attrs["n_models"]
                    prot_grp.attrs["L"]        = hf_in.attrs["L"]
                saved.append(pid)
                print(f"  [OK] {pid}")
            except Exception as e:
                print(f"  [FAIL] {pid}: {e}")
        hf_out.create_dataset(
            "protein_ids",
            data=np.array(saved, dtype="S"),
            compression="gzip")
    print(f"[MERGE] Done — {len(saved)} proteins saved")
    return saved
 
 
# ─────────────────────────── main ───────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_workers",  type=int,
                        default=min(mp.cpu_count(), 16))
    parser.add_argument("--timeout",    type=int, default=300)
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--keep_tmp",   action="store_true")
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir     = os.path.join(args.output_dir, "tmp")
    output_file = os.path.join(args.output_dir, "prepared_distograms.h5")
    os.makedirs(tmp_dir, exist_ok=True)
 
    df = pd.read_csv(args.input_csv)
    print(f"[INFO] {len(df)} proteins in CSV")
    print(f"[INFO] {args.n_workers} parallel workers  |  timeout={args.timeout}s/protein")
    print(f"[INFO] tmp dir  : {tmp_dir}")
    print(f"[INFO] output   : {output_file}")
 
    tasks         = []
    precheck_fail = []
 
    for _, row in df.iterrows():
        pid = str(row["protein_id"]).strip()
 
        try:
            pickle_files = [p.strip()
                            for p in str(row["pickle_files"]).split(";")
                            if p.strip()]
        except Exception as e:
            precheck_fail.append((pid, f"pickle_files parse error: {e}"))
            continue
 
        if not pickle_files:
            precheck_fail.append((pid, "empty pickle_files column"))
            continue
 
        task = {
            "pid":          pid,
            "pickle_files": pickle_files,
            "tmp_path":     os.path.join(tmp_dir, f"{pid}.h5"),
            "overwrite":    args.overwrite,
            "timeout":      args.timeout,
        }
 
        ok, warnings = check_task(task)
        for w in warnings:
            print(w)
        if not ok:
            precheck_fail.append((pid, "no valid pickle files after pre-check"))
            continue
 
        tasks.append(task)
 
    print(f"\n[INFO] {len(tasks)} proteins to process "
          f"({len(precheck_fail)} skipped at pre-check)\n")
 
    if not tasks:
        print("[INFO] Nothing to do.")
        return
 
    ok_pids   = []
    fail_list = []
    skip_list = []
    t0 = time.time()
 
    pool = mp.Pool(processes=args.n_workers)
    async_results = []
    for task in tasks:
        ar = pool.apply_async(process_protein, (task,))
        async_results.append((task["pid"], ar))
    pool.close()
 
    for i, (pid, ar) in enumerate(async_results, start=1):
        try:
            pid_result, status, msg = ar.get(timeout=args.timeout + 60)
        except mp.TimeoutError:
            status     = "fail"
            msg        = f"worker timeout after {args.timeout+60}s"
            pid_result = pid
        except Exception as e:
            status     = "fail"
            msg        = f"worker exception: {e}"
            pid_result = pid
 
        pct     = 100 * i / len(tasks)
        elapsed = time.time() - t0
        tag     = {"ok": "OK  ", "skip": "SKIP", "fail": "FAIL"}.get(status, "????")
        print(f"[{i:3d}/{len(tasks)}  {pct:5.1f}%  {elapsed:6.0f}s]  "
              f"{tag}  {pid_result}  — {msg}")
 
        if status in ("ok", "skip"):
            ok_pids.append(pid_result)
            if status == "skip":
                skip_list.append(pid_result)
        else:
            fail_list.append((pid_result, msg))
 
    pool.join()
 
    csv_order   = [str(r["protein_id"]).strip() for _, r in df.iterrows()]
    ok_pids_ord = [p for p in csv_order if p in set(ok_pids)]
    saved = merge(tmp_dir, output_file, ok_pids_ord)
 
    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("[INFO] tmp directory removed")
 
    all_fail = precheck_fail + fail_list
    print("\n" + "=" * 60)
    print(f"[SUMMARY]  Processed  : {len(ok_pids) - len(skip_list)}")
    print(f"[SUMMARY]  Skipped    : {len(skip_list)}")
    print(f"[SUMMARY]  Failed     : {len(all_fail)}")
    print(f"[SUMMARY]  Saved      : {len(saved)} proteins → {output_file}")
    print(f"[SUMMARY]  Total time : {time.time()-t0:.0f}s")
 
    if all_fail:
        fail_path = os.path.join(args.output_dir, "failed_proteins.txt")
        with open(fail_path, "w") as f:
            f.write("protein_id\treason\n")
            for pid, reason in all_fail:
                f.write(f"{pid}\t{reason}\n")
        print(f"[INFO] Failed list → {fail_path}")
 
        by_type = {}
        for _, reason in all_fail:
            key = reason.split(":")[0].split(" ")[0]
            by_type[key] = by_type.get(key, 0) + 1
        print("[INFO] Failure types:")
        for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"         {v:3d}x  {k}")
 
    print("=" * 60)
    print(output_file)
 
 
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
 
