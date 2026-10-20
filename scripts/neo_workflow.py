"""Plan, validate and summarize the two five-seed NEO GitHub workflows.

Standard library only. Does not run either optimizer or modify public data.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
import zipfile

from check_neo_solution import check_solution, read_sol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"runs": 5, "seed": 4321, "iter_k": 16, "max_levels": 4,
          "merge_ratio": 0.10, "tabu_factor": 0.25, "tabu1": 0.10, "tabu2": 0.25,
          "delta1": 0.3, "delta2": 0.2, "delta3": 0.1, "delta4": 0.3}
FINGERPRINT = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
REFERENCE_BASIS = "recomputed_public_routes"


def manifest(root):
    root = Path(root)
    table = root / "published.csv"
    if not table.exists():
        table = root / "manifest.csv"
    with table.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["instance"] for r in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate instance names")
    for row in rows:
        for field in ("instance_path", "solution_path", "json_path"):
            path = (root / row[field]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Invalid manifest path: " + str(path))
    return rows


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty result table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def public_reference(root, entry):
    """Recompute public routes; select the cheapest feasible alternative in a .sol."""
    root = Path(root)
    published = float(entry["objective"])
    solutions = read_sol(root / entry["solution_path"])
    result = {"instance": entry["instance"], "dataset": entry["dataset"],
              "customers": entry["customers"], "public_cost": published,
              "public_is_optimal_manifest": entry["is_optimal"],
              "reference_status": "REVIEW", "reference_recomputed": "",
              "comparison_basis": REFERENCE_BASIS, "public_solution_vrp_node_ids": "",
              "reference_solution_label": "", "reference_alternatives": len(solutions)}
    candidates = []
    for index, solution in enumerate(solutions):
        report = check_solution(root / entry["instance_path"], solution["routes"], published,
                                numbering="neo", has_depots=False)
        if report["feasible"] and report["cost_complete"]:
            candidates.append((report["computed_cost"], index, solution, report))
    result["reference_valid_alternatives"] = len(candidates)
    if candidates:
        cost, _, solution, report = min(candidates, key=lambda item: item[:2])
        result.update(reference_recomputed=cost, reference_status=report["status"],
                      reference_solution_label=solution["label"],
                      public_solution_vrp_node_ids=json.dumps(
                          [route["nodes"] for route in report["routes"]],
                          separators=(",", ":")))
    if entry["check"] != "PASS" or len(solutions) != 1:
        result["reference_status"] = "REVIEW"
    result["public_comparison_validated"] = "YES" if candidates else "NO"
    return result


def gap(cost, reference):
    if cost == "" or reference == "" or float(reference) <= 0:
        return ""
    return round(100 * (float(cost) - float(reference)) / float(reference), 6)


def summarize(algorithm, parts, output, dataset_root, sha="local", run_id="local"):
    entries = manifest(dataset_root)
    by_name = {entry["instance"]: entry for entry in entries}
    groups = defaultdict(list)
    for path in sorted(Path(parts).rglob("*.csv")):
        for row in read_csv(path):
            if row.get("instance") not in by_name:
                raise ValueError(f"Unexpected instance in {path}: {row.get('instance')}")
            groups[row["instance"]].append(row)
    all_runs, best_rows, errors = [], [], []
    for entry in entries:
        name = entry["instance"]
        ref = public_reference(dataset_root, entry)
        seeds = defaultdict(list)
        for row in groups[name]:
            seeds[int(row["seed"])].append(row)
        expected_seeds = range(CONFIG["seed"], CONFIG["seed"] + CONFIG["runs"])
        if set(seeds) - set(expected_seeds):
            raise ValueError(f"Unexpected seeds for {name}")
        valid = []
        for seed in expected_seeds:
            found = seeds.get(seed, [])
            if len(found) != 1:
                reason = "MISSING" if not found else "DUPLICATE_SEED"
                row = {"instance": name, "dataset": entry["dataset"], "run": seed - CONFIG["seed"] + 1,
                       "seed": seed, "algorithm": algorithm.upper(), "cost": "", "solution": "",
                       "validation_status": reason, "validation_error": reason}
            else:
                row = dict(found[0])
                try:
                    if row.get("algorithm") != algorithm.upper():
                        raise ValueError("Wrong algorithm label")
                    if int(row["run"]) != seed - CONFIG["seed"] + 1:
                        raise ValueError("Wrong run index")
                    for key in ("iter_k", "max_levels", "merge_ratio", "tabu_factor", "tabu1", "tabu2",
                                "delta1", "delta2", "delta3", "delta4"):
                        if float(row[key]) != CONFIG[key]:
                            raise ValueError("Wrong parameter: " + key)
                    budget = int(entry["customers"]) * CONFIG["iter_k"] * (2 * CONFIG["max_levels"] + 1)
                    if int(row["customers"]) != int(entry["customers"]):
                        raise ValueError("Wrong customer count")
                    if int(row["iteration_budget"]) != budget or not 0 <= int(row["iterations_used"]) <= budget:
                        raise ValueError("Wrong iteration budget/counter")
                    report = check_solution(Path(dataset_root) / entry["instance_path"],
                                            json.loads(row["solution"]), float(row["cost"]))
                    if report["status"] != "PASS" or row["feasible"] != "YES":
                        raise ValueError("Route/cost validation failed: " + "; ".join(report["errors"]))
                    if not math.isfinite(float(row["runtime_s"])) or float(row["runtime_s"]) < 0:
                        raise ValueError("Invalid runtime")
                    row.update(validation_status="PASS", validation_error="")
                    valid.append(row)
                except (ValueError, KeyError, TypeError) as exc:
                    row.update(validation_status="INVALID", validation_error=str(exc))
            row.update(ref)
            row["gap_pct"] = gap(row.get("cost", ""), ref["reference_recomputed"]) if row["validation_status"] == "PASS" else ""
            row["gap_recomputed_pct"] = row["gap_pct"]
            all_runs.append(row)
            if row["validation_status"] != "PASS":
                errors.append(f"{name} seed={seed}: {row['validation_error']}")
        summary = dict(ref, expected_runs=CONFIG["runs"], rows_found=len(groups[name]),
                       valid_runs=len(valid), complete="YES" if len(valid) == CONFIG["runs"] else "NO",
                       best_cost="", mean_cost="", worst_cost="", stddev_cost="", best_seed="",
                       best_runtime_s="", mean_runtime_s="", best_solution="", iteration_budget="",
                       best_iterations_used="", gap_public_pct="", gap_recomputed_pct="")
        if valid:
            best = min(valid, key=lambda r: (float(r["cost"]), int(r["seed"])))
            values = [float(r["cost"]) for r in valid]
            summary.update(best_cost=float(best["cost"]), mean_cost=round(statistics.mean(values), 6),
                           worst_cost=max(values), stddev_cost=round(statistics.pstdev(values), 6),
                           best_seed=best["seed"], best_runtime_s=best["runtime_s"],
                           mean_runtime_s=round(statistics.mean(float(r["runtime_s"]) for r in valid), 6),
                           best_solution=best["solution"], iteration_budget=best["iteration_budget"],
                           best_iterations_used=best["iterations_used"], gap_public_pct=gap(best["cost"], ref["reference_recomputed"]),
                           gap_recomputed_pct=gap(best["cost"], ref["reference_recomputed"]))
        best_rows.append(summary)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", all_runs)
    write_csv(out / "best.csv", best_rows)
    write_csv(out / "public_reference.csv", [public_reference(dataset_root, e) for e in entries])
    metadata = {"algorithm": algorithm, "sha": sha, "run_id": str(run_id), "config": CONFIG,
                "config_fingerprint": FINGERPRINT, "comparison_basis": REFERENCE_BASIS, "instances": len(entries),
                "expected_runs": len(entries) * CONFIG["runs"],
                "valid_runs": sum(r["validation_status"] == "PASS" for r in all_runs), "complete": not errors}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "validation_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    message = f"{algorithm.upper()}: {metadata['valid_runs']}/{metadata['expected_runs']} validated runs; best.csv has {len(best_rows)} instances."
    print(message)
    step_summary("### NEO " + message + "\n")
    if errors:
        print("\n".join(errors[:20]))
    return metadata


def load_bundle(path):
    path = Path(path)
    return (json.loads((path / "metadata.json").read_text(encoding="utf-8")),
            read_csv(path / "best.csv"), read_csv(path / "runs.csv"))


def compare(ml_dir, tabu_dir, output):
    ml_meta, ml_best, ml_runs = load_bundle(ml_dir)
    tabu_meta, tabu_best, tabu_runs = load_bundle(tabu_dir)
    if not ml_meta["complete"] or not tabu_meta["complete"]:
        raise ValueError("Cannot compare as best-of-five: a workflow has missing/invalid runs")
    if ml_meta["sha"] != tabu_meta["sha"] or ml_meta["config"] != tabu_meta["config"]:
        raise ValueError("Refuse to mix different source commits/configurations")
    if ml_meta["algorithm"] != "ml" or tabu_meta["algorithm"] != "tabu":
        raise ValueError("Wrong algorithm bundle")
    if any(meta.get("comparison_basis") != REFERENCE_BASIS for meta in (ml_meta, tabu_meta)):
        raise ValueError("Regenerate summaries using recomputed public route costs")
    ml = {r["instance"]: r for r in ml_best}
    tabu = {r["instance"]: r for r in tabu_best}
    if set(ml) != set(tabu):
        raise ValueError("Different instance sets")
    paired = {}
    for name in ml:
        left = {int(r["seed"]): r for r in ml_runs if r["instance"] == name}
        right = {int(r["seed"]): r for r in tabu_runs if r["instance"] == name}
        if set(left) != set(right) or len(left) != CONFIG["runs"]:
            raise ValueError("Different seed sets: " + name)
        if any(int(left[s]["iteration_budget"]) != int(right[s]["iteration_budget"]) for s in left):
            raise ValueError("Different budgets: " + name)
        paired[name] = all(int(left[s]["iterations_used"]) == int(right[s]["iterations_used"]) for s in left)
    result = []
    for name in sorted(ml):
        a, b = ml[name], tabu[name]
        if any(a[key] != b[key] for key in ("reference_recomputed", "public_solution_vrp_node_ids")):
            raise ValueError("Different recomputed public references: " + name)
        ac, bc = float(a["best_cost"]), float(b["best_cost"])
        row = {"instance": name, "dataset": a["dataset"], "customers": a["customers"],
               "public_cost": a["public_cost"], "reference_recomputed": a["reference_recomputed"],
               "reference_status": a["reference_status"], "public_comparison_validated": a["public_comparison_validated"],
               "public_is_optimal_manifest": a["public_is_optimal_manifest"],
               "comparison_basis": REFERENCE_BASIS,
               "reference_solution_label": a["reference_solution_label"],
               "reference_alternatives": a["reference_alternatives"],
               "ML_best": ac, "Tabu_best": bc,
               "ML_gap_public_pct": gap(ac, a["reference_recomputed"]), "Tabu_gap_public_pct": gap(bc, a["reference_recomputed"]),
               "ML_gap_recomputed_pct": gap(ac, a["reference_recomputed"]), "Tabu_gap_recomputed_pct": gap(bc, a["reference_recomputed"]),
               "ML_vs_Tabu_gap_pct": gap(ac, bc),
               "winner": "ML" if ac < bc else "TABU" if ac > bc else "TIE",
               "ML_best_seed": a["best_seed"], "Tabu_best_seed": b["best_seed"],
               "ML_valid_runs": a["valid_runs"], "Tabu_valid_runs": b["valid_runs"],
               "ML_mean": a["mean_cost"], "Tabu_mean": b["mean_cost"],
               "ML_mean_runtime_s": a["mean_runtime_s"], "Tabu_mean_runtime_s": b["mean_runtime_s"],
               "iteration_budget": a["iteration_budget"], "same_seed_set": "YES", "same_budget": "YES",
               "paired_actual_iterations_equal": "YES" if paired[name] else "NO",
               "ML_best_iterations_used": a["best_iterations_used"], "Tabu_best_iterations_used": b["best_iterations_used"],
               "ML_best_solution_vrp_node_ids": a["best_solution"], "Tabu_best_solution_vrp_node_ids": b["best_solution"],
               "public_solution_vrp_node_ids": a["public_solution_vrp_node_ids"],
               "source_sha": ml_meta["sha"], "ML_workflow_run_id": ml_meta["run_id"], "Tabu_workflow_run_id": tabu_meta["run_id"]}
        result.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ML_vs_Tabu_vs_public.csv", result)
    validated = [r for r in result if r["public_comparison_validated"] == "YES"]
    wins = Counter(r["winner"] for r in result)
    lines = ["## NEO best of 5: ML vs Tabu vs public",
             f"Instances: {len(result)}; ML wins: {wins['ML']}; Tabu wins: {wins['TABU']}; ties: {wins['TIE']}.",
             "Gap (%) = 100 * (algorithm best - recomputed public route cost) / recomputed public route cost.",
             f"Feasible recomputed public references: {len(validated)}/{len(result)}."]
    if validated:
        lines.append(f"Mean instance gap on validated references: ML {statistics.mean(r['ML_gap_public_pct'] for r in validated):.4f}%; Tabu {statistics.mean(r['Tabu_gap_public_pct'] for r in validated):.4f}%.")
    lines.append("Published cost is retained for audit only. COST_MISMATCH/REVIEW references are included when their selected routes are feasible and their cost can be recomputed; missing/invalid references have blank gaps. For multiple public solutions, select the cheapest feasible alternative.")
    if any(not paired[name] for name in paired):
        lines.append("Some paired runs used fewer actual iterations despite equal ceilings; see paired_actual_iterations_equal.")
    (output / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    step_summary("\n\n".join(lines))
    print("\n".join(lines))
    return result


def step_summary(text):
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")


class NoCrossHostAuth(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def api(url, token, binary=False):
    req = Request(url, headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                                "X-GitHub-Api-Version": "2022-11-28"})
    with build_opener(NoCrossHostAuth()).open(req, timeout=30) as response:
        data = response.read()
    return data if binary else json.loads(data)


def fetch_bundle(algorithm, target, repository, sha, branch, token):
    endpoint = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    query = urlencode({"head_sha": sha, "branch": branch, "per_page": 20})
    try:
        runs = api(f"{endpoint}/repos/{repository}/actions/workflows/run_neo_{algorithm}.yml/runs?{query}", token)["workflow_runs"]
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    for run in runs:
        if run["head_sha"] != sha:
            continue
        artifacts = api(f"{endpoint}/repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100", token)["artifacts"]
        for artifact in artifacts:
            if artifact["name"] != f"neo-{algorithm}-results" or artifact["expired"]:
                continue
            try:
                archive = api(f"{endpoint}/repos/{repository}/actions/artifacts/{artifact['id']}/zip", token, binary=True)
            except HTTPError as exc:
                if exc.code in (404, 410):
                    continue
                raise
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                # Read only known CSV/JSON members. Never execute/extract artifact paths.
                names = {Path(n).name: n for n in bundle.namelist()}
                if not all(n in names for n in ("metadata.json", "runs.csv", "best.csv")):
                    continue
                meta = json.loads(bundle.read(names["metadata.json"]))
                if (not meta.get("complete") or meta.get("sha") != sha or meta.get("algorithm") != algorithm
                        or meta.get("config") != CONFIG or meta.get("config_fingerprint") != FINGERPRINT
                        or meta.get("comparison_basis") != REFERENCE_BASIS):
                    continue
                target = Path(target)
                target.mkdir(parents=True, exist_ok=True)
                for name in ("metadata.json", "runs.csv", "best.csv"):
                    (target / name).write_bytes(bundle.read(names[name]))
                print(f"Found {algorithm.upper()} artifact from workflow run {run['id']} at {sha}")
                return target
    return None


def compare_remote(algorithm, current_dir, output, repository, sha, branch):
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        raise ValueError("GH_TOKEN is required to read workflow artifacts")
    current = Path(current_dir)
    peer_algorithm = "tabu" if algorithm == "ml" else "ml"
    peer = Path(output) / "peer"
    for attempt in range(3):
        if not (current / "metadata.json").exists():
            fetch_bundle(algorithm, current, repository, sha, branch, token)
        other = fetch_bundle(peer_algorithm, peer, repository, sha, branch, token)
        if (current / "metadata.json").exists() and other:
            meta, _, _ = load_bundle(current)
            if meta["sha"] != sha or meta["config"] != CONFIG or not meta["complete"]:
                raise ValueError("Current result bundle is incomplete or from another commit/config")
            compare(current if algorithm == "ml" else other, other if algorithm == "ml" else current, output)
            return True
        if attempt < 2:
            time.sleep(5)
    message = ("Comparison pending: no complete results for BOTH algorithms at this exact commit/config. "
               "Run the other NEO workflow on the same commit. The workflow finishing later will compare; "
               "or dispatch either workflow with mode=compare-only to retry without rerunning the solvers.")
    print(message)
    step_summary("### " + message)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--dataset-root", type=Path, default=ROOT / "NEO_CVRP_62_pairs")
    summary = sub.add_parser("summarize")
    summary.add_argument("--algorithm", choices=("ml", "tabu"), required=True)
    summary.add_argument("--parts", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--dataset-root", type=Path, default=ROOT / "NEO_CVRP_62_pairs")
    summary.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "local"))
    summary.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    local = sub.add_parser("compare")
    local.add_argument("--ml", required=True)
    local.add_argument("--tabu", required=True)
    local.add_argument("--output", required=True)
    remote = sub.add_parser("compare-remote")
    remote.add_argument("--algorithm", choices=("ml", "tabu"), required=True)
    remote.add_argument("--current-dir", required=True)
    remote.add_argument("--output", required=True)
    remote.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    remote.add_argument("--sha", default=os.environ.get("GITHUB_SHA"))
    remote.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME"))
    args = parser.parse_args()
    try:
        if args.command == "plan":
            rows = manifest(args.dataset_root)
            if len(rows) > 256:
                raise ValueError("Too many instances for one GitHub matrix")
            matrix = {"include": [{"instance": r["instance"], "path": (Path(args.dataset_root) / r["instance_path"]).relative_to(ROOT).as_posix()} for r in rows]}
            print(json.dumps(matrix, separators=(",", ":")))
        elif args.command == "summarize":
            return 0 if summarize(args.algorithm, args.parts, args.output, args.dataset_root, args.sha, args.run_id)["complete"] else 1
        elif args.command == "compare":
            compare(args.ml, args.tabu, args.output)
        else:
            if not args.repository or not args.sha or not args.branch:
                raise ValueError("Repository, SHA and branch are required")
            compare_remote(args.algorithm, args.current_dir, args.output, args.repository, args.sha, args.branch)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print("ERROR:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
