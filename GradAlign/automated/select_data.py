#!/usr/bin/env python3

import os
import json
import argparse
import random
import math
import hashlib
from datetime import datetime, timezone
from typing import List, Set, Dict, Tuple, Optional

import datasets as hfds  # type: ignore[import-not-found]

from config import get_dataset_dir, get_response_dir


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str, payload: Dict) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _atomic_write_jsonl(path: str, rows: List[Dict]) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _read_topn_group_ids(sim_jsonl_path: str, top_n: int, neg: bool = False) -> List[int]:
    pairs: List[Tuple[int, float]] = []
    with open(sim_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "group_id" in obj and "similarity" in obj:
                try:
                    gid = int(obj["group_id"])
                    sim = float(obj["similarity"])
                except (TypeError, ValueError):
                    continue
                pairs.append((gid, sim))
    pairs.sort(key=lambda x: x[1], reverse=not neg)
    # print('n', top_n)
    print([x[1] for x in pairs[top_n-10:top_n]])
    return [gid for gid, _ in pairs[:top_n]]


def _read_similarity_map(sim_jsonl_path: str) -> Dict[int, float]:
    sim_map: Dict[int, float] = {}
    with open(sim_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "group_id" in obj and "similarity" in obj:
                try:
                    gid = int(obj["group_id"])
                    sim = float(obj["similarity"])
                except (TypeError, ValueError):
                    continue
                sim_map[gid] = sim
    return sim_map


def _read_svd_score_data(
    svd_jsonl_path: str,
) -> Tuple[Dict[int, float], Optional[str], Optional[str]]:
    """Read the final per-prompt S from effective_rank_topk.s."""
    score_map: Dict[int, float] = {}
    analysis_signatures: Set[str] = set()
    score_scopes: Set[str] = set()
    with open(svd_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"Invalid JSON in SVD score file {svd_jsonl_path} at line {line_num}"
                ) from error
            if "group_id" not in obj:
                continue
            try:
                gid = int(obj["group_id"])
                score = float(obj["effective_rank_topk"]["s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Missing or invalid effective_rank_topk.s in {svd_jsonl_path} "
                    f"at line {line_num}; do not mix old SVD records without S scores"
                ) from exc
            if not math.isfinite(score):
                raise SystemExit(
                    f"Non-finite effective_rank_topk.s for group_id {gid} "
                    f"in {svd_jsonl_path} at line {line_num}"
                )
            if gid in score_map:
                raise SystemExit(
                    f"Duplicate group_id {gid} in SVD score file {svd_jsonl_path} "
                    f"(line {line_num})"
                )
            score_map[gid] = score
            signature = obj.get("analysis_signature")
            if isinstance(signature, str) and signature:
                analysis_signatures.add(signature)
            score_scope = obj.get("svd_score_scope")
            score_record = obj.get("effective_rank_topk")
            if not score_scope and isinstance(score_record, dict):
                score_scope = score_record.get("score_scope")
            if not score_scope:
                score_scope = obj.get("gradient_parameter_scope")
            if isinstance(score_scope, str) and score_scope:
                score_scopes.add(score_scope)
    if len(analysis_signatures) > 1:
        raise SystemExit(
            f"Mixed analysis signatures in SVD score file {svd_jsonl_path}"
        )
    if len(score_scopes) > 1:
        raise SystemExit(f"Mixed SVD score scopes in {svd_jsonl_path}")
    signature = next(iter(analysis_signatures), None)
    score_scope = next(iter(score_scopes), None)
    return score_map, signature, score_scope


def _read_svd_score_map(svd_jsonl_path: str) -> Dict[int, float]:
    """Backward-compatible score-only reader."""
    return _read_svd_score_data(svd_jsonl_path)[0]


def _read_subspace_score_data(
    score_path: str,
) -> Tuple[Dict[int, float], Optional[str], Optional[str], Optional[str]]:
    """Read the per-prompt normalized U/V subspace selection score."""
    score_map: Dict[int, float] = {}
    signatures: Set[str] = set()
    scopes: Set[str] = set()
    sides: Set[str] = set()
    with open(score_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                group_id = int(row["group_id"])
                record = row["subspace_similarity"]
                score = float(record["s"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise SystemExit(
                    f"Invalid subspace score in {score_path}:{line_number}"
                ) from error
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise SystemExit(
                    f"Out-of-range subspace score for group_id={group_id}: {score}"
                )
            if group_id in score_map:
                raise SystemExit(
                    f"Duplicate group_id={group_id} in {score_path}:{line_number}"
                )
            score_map[group_id] = score
            signature = row.get("analysis_signature")
            scope = record.get("score_scope")
            side = record.get("score_side")
            if isinstance(signature, str) and signature:
                signatures.add(signature)
            if isinstance(scope, str) and scope:
                scopes.add(scope)
            if isinstance(side, str) and side:
                sides.add(side)
    if len(signatures) > 1 or len(scopes) > 1 or len(sides) > 1:
        raise SystemExit(f"Mixed subspace configurations in {score_path}")
    return (
        score_map,
        next(iter(signatures), None),
        next(iter(scopes), None),
        next(iter(sides), None),
    )


def _read_candidate_group_ids(dataset_train_path: str) -> List[int]:
    group_ids: List[int] = []
    seen: Set[int] = set()
    with open(dataset_train_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                group_id = int(row["extra_info"]["index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise SystemExit(
                    f"Invalid candidate group_id at {dataset_train_path}:{line_number}"
                ) from error
            if group_id in seen:
                raise SystemExit(
                    f"Duplicate candidate group_id={group_id} in {dataset_train_path}"
                )
            seen.add(group_id)
            group_ids.append(group_id)
    if not group_ids:
        raise SystemExit(f"Candidate dataset is empty: {dataset_train_path}")
    return group_ids


def _read_acc_map(acc_jsonl_path: str) -> Dict[int, float]:
    acc: Dict[int, float] = {}
    with open(acc_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "group_id" in obj and "accuracy" in obj:
                try:
                    gid = int(obj["group_id"])
                    a = float(obj["accuracy"])
                except (TypeError, ValueError):
                    continue
                acc[gid] = a
    return acc


def _write_selected(dataset_train_path: str, selected_ids: Set[int], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "train.jsonl")
    selected: List[Dict] = []
    with open(dataset_train_path, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)
            idx = int(obj['extra_info']['index'])
            if idx in selected_ids:
                selected.append(obj)
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out_parquet = out_jsonl.replace(".jsonl", ".parquet")
    if selected:
        hfds.Dataset.from_list(selected).to_parquet(out_parquet)
    return out_jsonl, out_parquet


def select_by_similarity(dataset_dir: str, parts_root: str, top_n: int, output_dir: str, filename: str, neg: bool = False) -> None:
    sim_path = os.path.join(parts_root, filename)
    assert os.path.isfile(sim_path), f"Similarity file not found: {sim_path}"
    gids = _read_topn_group_ids(sim_path, top_n, neg)
    selected_ids = set(gids)
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_by_svd_score(
    dataset_dir: str,
    parts_root: str,
    top_n: int,
    output_dir: str,
    svd_rank: int,
    iteration: Optional[int] = None,
    global_step: Optional[int] = None,
) -> None:
    score_path = os.path.join(parts_root, f"svd_results_top{svd_rank}_aggregated.jsonl")
    if not os.path.isfile(score_path):
        raise SystemExit(f"Aggregated SVD score file not found: {score_path}")

    score_map, analysis_signature, svd_score_scope = _read_svd_score_data(score_path)
    if not score_map:
        raise SystemExit(
            f"No effective_rank_topk.s values found in {score_path}; "
            "rerun SVD analysis with the effective-rank score implementation"
        )

    if top_n <= 0:
        raise SystemExit("SVD selection count must be positive")
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    candidate_group_ids = _read_candidate_group_ids(dataset_train)
    candidate_set = set(candidate_group_ids)
    scored_set = set(score_map)
    if candidate_set != scored_set:
        missing_scores = sorted(candidate_set.difference(scored_set))
        unknown_scores = sorted(scored_set.difference(candidate_set))
        raise SystemExit(
            "Candidate/SVD group_id mismatch: "
            f"candidates={len(candidate_set)}, scored={len(scored_set)}, "
            f"missing_scores={len(missing_scores)} {missing_scores[:10]}, "
            f"unknown_scores={len(unknown_scores)} {unknown_scores[:10]}"
        )

    ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    requested_top_n = top_n
    if top_n > len(ordered):
        print(f"Warning: requested {top_n} but only {len(ordered)} SVD scores are available; selecting all.")
        top_n = len(ordered)
    boundary_start = max(0, top_n - 10)
    print("S scores at selection boundary:", [score for _, score in ordered[boundary_start:top_n]])
    selected_ids = {gid for gid, _ in ordered[:top_n]}
    print("Selected", len(selected_ids), "prompts by descending effective-rank S")
    selected_jsonl, selected_parquet = _write_selected(
        dataset_train, selected_ids, output_dir
    )
    selected_rows = _read_candidate_group_ids(selected_jsonl)
    if set(selected_rows) != selected_ids or len(selected_rows) != len(selected_ids):
        raise SystemExit("Selected dataset does not match the ranked group_id set")

    selection_score_rows = [
        {
            "group_id": group_id,
            "score": score,
            "rank": rank,
            "selected": rank <= top_n,
        }
        for rank, (group_id, score) in enumerate(ordered, 1)
    ]
    selection_scores_path = os.path.join(output_dir, "selection_scores.jsonl")
    _atomic_write_jsonl(selection_scores_path, selection_score_rows)
    selected_group_ids = [group_id for group_id, _ in ordered[:top_n]]
    manifest = {
        "selection_schema_version": 1,
        "selection_method": "svd_effective_rank_topk_descending",
        "iteration": iteration,
        "global_step": global_step,
        "candidate_count": len(candidate_group_ids),
        "scored_candidate_count": len(score_map),
        "requested_selected_count": requested_top_n,
        "selected_count": len(selected_ids),
        "score_cutoff": float(ordered[top_n - 1][1]),
        "score_min": float(min(score_map.values())),
        "score_max": float(max(score_map.values())),
        "selected_group_ids": selected_group_ids,
        "analysis_signature": analysis_signature,
        "svd_rank": svd_rank,
        "svd_score_scope": svd_score_scope,
        "candidate_jsonl": os.path.abspath(dataset_train),
        "candidate_jsonl_sha256": _sha256_file(dataset_train),
        "source_scores_jsonl": os.path.abspath(score_path),
        "source_scores_sha256": _sha256_file(score_path),
        "selection_scores_jsonl": os.path.abspath(selection_scores_path),
        "selection_scores_sha256": _sha256_file(selection_scores_path),
        "selected_jsonl": os.path.abspath(selected_jsonl),
        "selected_jsonl_sha256": _sha256_file(selected_jsonl),
        "selected_parquet": os.path.abspath(selected_parquet),
        "selected_parquet_sha256": _sha256_file(selected_parquet),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(output_dir, "selection_manifest.json")
    _atomic_write_json(manifest_path, manifest)
    print(f"Selection scores written to: {selection_scores_path}")
    print(f"Selection manifest written to: {manifest_path}")


def select_by_subspace_score(
    dataset_dir: str,
    parts_root: str,
    top_n: int,
    output_dir: str,
    svd_rank: int,
    iteration: Optional[int] = None,
    global_step: Optional[int] = None,
) -> None:
    score_path = os.path.join(
        parts_root,
        f"subspace_results_top{svd_rank}_aggregated.jsonl",
    )
    if not os.path.isfile(score_path):
        raise SystemExit(f"Aggregated subspace score file not found: {score_path}")
    score_map, signature, score_scope, score_side = _read_subspace_score_data(
        score_path
    )
    if not score_map or top_n <= 0:
        raise SystemExit("Subspace selection needs scores and a positive count")

    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    candidate_group_ids = _read_candidate_group_ids(dataset_train)
    candidate_set = set(candidate_group_ids)
    scored_set = set(score_map)
    if candidate_set != scored_set:
        missing_scores = sorted(candidate_set.difference(scored_set))
        unknown_scores = sorted(scored_set.difference(candidate_set))
        raise SystemExit(
            "Candidate/subspace group_id mismatch: "
            f"candidates={len(candidate_set)}, scored={len(scored_set)}, "
            f"missing_scores={len(missing_scores)} {missing_scores[:10]}, "
            f"unknown_scores={len(unknown_scores)} {unknown_scores[:10]}"
        )

    ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    requested_top_n = top_n
    top_n = min(top_n, len(ordered))
    boundary_start = max(0, top_n - 10)
    print(
        "Subspace scores at selection boundary:",
        [score for _, score in ordered[boundary_start:top_n]],
    )
    selected_ids = {group_id for group_id, _ in ordered[:top_n]}
    print(
        f"Selected {len(selected_ids)} prompts by descending phi_{score_side} "
        f"subspace score"
    )
    selected_jsonl, selected_parquet = _write_selected(
        dataset_train, selected_ids, output_dir
    )
    selected_rows = _read_candidate_group_ids(selected_jsonl)
    if set(selected_rows) != selected_ids or len(selected_rows) != len(selected_ids):
        raise SystemExit("Selected dataset does not match subspace ranking")

    selection_score_rows = [
        {
            "group_id": group_id,
            "score": score,
            "rank": rank,
            "selected": rank <= top_n,
        }
        for rank, (group_id, score) in enumerate(ordered, 1)
    ]
    selection_scores_path = os.path.join(output_dir, "selection_scores.jsonl")
    _atomic_write_jsonl(selection_scores_path, selection_score_rows)
    manifest = {
        "selection_schema_version": 1,
        "selection_method": "adamw_backbone_subspace_topk_descending",
        "iteration": iteration,
        "global_step": global_step,
        "candidate_count": len(candidate_group_ids),
        "scored_candidate_count": len(score_map),
        "requested_selected_count": requested_top_n,
        "selected_count": len(selected_ids),
        "score_cutoff": float(ordered[top_n - 1][1]),
        "score_min": float(min(score_map.values())),
        "score_max": float(max(score_map.values())),
        "selected_group_ids": [group_id for group_id, _ in ordered[:top_n]],
        "analysis_signature": signature,
        "svd_rank": svd_rank,
        "svd_score_scope": score_scope,
        "subspace_score_side": score_side,
        "candidate_jsonl": os.path.abspath(dataset_train),
        "candidate_jsonl_sha256": _sha256_file(dataset_train),
        "source_scores_jsonl": os.path.abspath(score_path),
        "source_scores_sha256": _sha256_file(score_path),
        "selection_scores_jsonl": os.path.abspath(selection_scores_path),
        "selection_scores_sha256": _sha256_file(selection_scores_path),
        "selected_jsonl": os.path.abspath(selected_jsonl),
        "selected_jsonl_sha256": _sha256_file(selected_jsonl),
        "selected_parquet": os.path.abspath(selected_parquet),
        "selected_parquet_sha256": _sha256_file(selected_parquet),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(output_dir, "selection_manifest.json")
    _atomic_write_json(manifest_path, manifest)
    print(f"Selection scores written to: {selection_scores_path}")
    print(f"Selection manifest written to: {manifest_path}")


def select_by_accuracy(dataset_dir: str, parts_root: str, n: int, acc_low: float, acc_high: float, output_dir: str, lim: int | None = None) -> None:
    acc_path = os.path.join(parts_root, "accuracy_by_problem.jsonl")
    assert os.path.isfile(acc_path), f"Accuracy file not found: {acc_path}"
    acc_map = _read_acc_map(acc_path)
    # print(acc_map)
    pool = [gid for gid, a in acc_map.items() if acc_low <= a <= acc_high and (lim is None or gid < int(lim))]
    if not pool:
        raise SystemExit("No group_ids satisfy the accuracy constraints")
    if n > len(pool):
        print(f"Warning: requested {n} but only {len(pool)} available; selecting all.")
        n = len(pool)
    selected_ids = set(random.sample(pool, n))
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_by_simacc(dataset_dir: str, parts_root: str, n: int, output_dir: str, filename: str) -> None:
    sim_path = os.path.join(parts_root, filename)
    acc_path = os.path.join(parts_root, "accuracy_by_problem.jsonl")
    assert os.path.isfile(sim_path), f"Similarity file not found: {sim_path}"
    assert os.path.isfile(acc_path), f"Accuracy file not found: {acc_path}"
    sim_map = _read_similarity_map(sim_path)
    acc_map = _read_acc_map(acc_path)
    pairs: List[Tuple[int, float, float, float]] = []
    for gid, sim in sim_map.items():
        if gid not in acc_map:
            continue
        a = acc_map[gid]
        score = sim * (a * (1.0 - a)) * 4
        pairs.append((gid, score, sim, a))
    if not pairs:
        raise SystemExit("No overlapping group_ids between similarity and accuracy inputs")
    pairs.sort(key=lambda x: x[1], reverse=True)
    if n > len(pairs):
        print(f"Warning: requested {n} but only {len(pairs)} available; selecting all.")
        n = len(pairs)
    print(pairs[n-10:n])
    selected_ids = set(gid for gid, _, _, _ in pairs[:n])
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_by_accgreedy(dataset_dir: str, parts_root: str, n: int, output_dir: str) -> None:
    acc_path = os.path.join(parts_root, "accuracy_by_problem.jsonl")
    assert os.path.isfile(acc_path), f"Accuracy file not found: {acc_path}"
    acc_map = _read_acc_map(acc_path)
    if not acc_map:
        raise SystemExit("No accuracy records found for accgreedy selection")
    # Sort by closeness to 0.5
    ordered = sorted(acc_map.items(), key=lambda kv: abs(kv[1] - 0.5))
    if n > len(ordered):
        print(f"Warning: requested {n} but only {len(ordered)} available; selecting all.")
        n = len(ordered)
    selected_ids = set(gid for gid, _ in ordered[:n])
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_random(dataset_dir: str, n: int, output_dir: str, max_num: int | None = None) -> None:
    # Count lines up to max_num (if provided)
    total = 0
    with open(os.path.join(dataset_dir, "train.jsonl"), "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
            if max_num is not None and total >= int(max_num):
                break
    if n > total:
        print(f"Warning: requested {n} but only {total} available; selecting all.")
        n = total
    selected_indices = set(sorted(random.sample(range(total), n)))
    print('Selected', len(selected_indices))
    _write_selected(os.path.join(dataset_dir, "train.jsonl"), selected_indices, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Unified data selector by similarity/SVD/accuracy/random")
    parser.add_argument("--mode", required=True, choices=["sim", "svd", "subspace", "acc", "rand", 'simacc', 'accgreedy', 'align', 'negsim'], help="Selection mode")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset key")
    parser.add_argument("--model", required=True, type=str, help="Tokenizer/model key used for dataset preparation")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        type=str,
        help=(
            "Explicit directory containing the candidate train.jsonl. "
            "Use this for dynamic selection so scores are mapped back to the "
            "same chunk that produced the responses."
        ),
    )
    parser.add_argument("--n", type=int, default=10000, help="Number of items to select (used in all modes, replaces --top_n)")
    parser.add_argument("--acc_low", type=float, default=0.2)
    parser.add_argument("--acc_high", type=float, default=0.8)
    parser.add_argument("--lim", type=int, default=None, help="Optional limit on candidates (group_id upper bound in acc; max lines in rand)")
    parser.add_argument("--similarity_filename", type=str, default="similarity_results_aggregated.jsonl")
    parser.add_argument("--svd_rank", type=int, default=128,
                        help="Top-k rank used in the aggregated SVD score filename")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; rand mode uses seed + iteration",
    )
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--global_step", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Optional explicit output directory")
    parser.add_argument("--parts_root", type=str, default=None, help="Override responses split root (compat with dynamic selection)")

    args = parser.parse_args()

    dataset_dir = (
        os.path.abspath(os.path.expanduser(args.dataset_dir))
        if args.dataset_dir
        else get_dataset_dir(args.dataset, args.model)
    )
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    if not os.path.isfile(dataset_train):
        raise SystemExit(f"Candidate dataset train.jsonl not found: {dataset_train}")
    if args.output_dir is None:
        # Place selections under dataset_dir/selected/<mode-specific>
        selected_root = os.path.join(dataset_dir, "selected", args.model)
        if args.mode == "sim":
            subdir = f"sim_{args.n}"
        elif args.mode == "svd":
            subdir = f"svd_s_top{args.svd_rank}_{args.n}"
        elif args.mode == "subspace":
            subdir = f"subspace_top{args.svd_rank}_{args.n}"
        elif args.mode == "align":
            subdir = f"align_{args.n}"
        elif args.mode == "negsim":
            subdir = f"negsim_{args.n}"
        elif args.mode == "simacc":
            subdir = f"simacc_{args.n}"
        elif args.mode == "acc":
            subdir = f"acc_{args.acc_low}_{args.acc_high}_{args.n}"
        elif args.mode == "accgreedy":
            subdir = f"accgreedy_{args.n}"
        else:
            subdir = f"random_{args.n}"
        output_dir = os.path.join(selected_root, subdir)
    else:
        output_dir = args.output_dir

    parts_root = None
    if args.mode in {"sim", "svd", "subspace", "acc", "simacc", "accgreedy", "align", "negsim"}:
        if not args.model:
            raise SystemExit("--model is required for sim/acc modes to locate responses")
        parts_root = args.parts_root or (get_response_dir(args.dataset, args.model) + "_split")
        if not os.path.isdir(parts_root):
            raise SystemExit(f"Parts root not found: {parts_root}")

    if args.mode == "sim" or args.mode == "align" or args.mode == "negsim":
        assert parts_root is not None
        select_by_similarity(dataset_dir, parts_root, int(args.n), output_dir, args.similarity_filename, neg=args.mode == "negsim")
    elif args.mode == "svd":
        assert parts_root is not None
        select_by_svd_score(
            dataset_dir,
            parts_root,
            int(args.n),
            output_dir,
            args.svd_rank,
            iteration=args.iteration,
            global_step=args.global_step,
        )
    elif args.mode == "subspace":
        assert parts_root is not None
        select_by_subspace_score(
            dataset_dir,
            parts_root,
            int(args.n),
            output_dir,
            args.svd_rank,
            iteration=args.iteration,
            global_step=args.global_step,
        )
    elif args.mode == "simacc":
        assert parts_root is not None
        select_by_simacc(dataset_dir, parts_root, int(args.n), output_dir, args.similarity_filename)
    elif args.mode == "acc":
        assert parts_root is not None
        select_by_accuracy(dataset_dir, parts_root, int(args.n), float(args.acc_low), float(args.acc_high), output_dir, lim=args.lim)
    elif args.mode == "accgreedy":
        assert parts_root is not None
        select_by_accgreedy(dataset_dir, parts_root, int(args.n), output_dir)
    else:
        iteration = args.iteration if args.iteration is not None else 0
        selection_seed = args.seed + iteration
        random.seed(selection_seed)
        print(
            f"Random selection seed: {selection_seed} "
            f"(base={args.seed}, iteration={iteration})"
        )
        select_random(dataset_dir, int(args.n), output_dir, max_num=args.lim)

    print(f"Selection complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
