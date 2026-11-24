"""Compare feasible makespan results from the Tabu/Multilevel workflows."""
import argparse
from collections import Counter, defaultdict
import csv
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

SOURCES = {
    "50_100": {
        "tabu": ("run_tabu.yml", "tabu-results", "TABU_RESULTS.csv"),
        "ml": ("run_multilevel_tabu.yml", "multilevel-tabu-results", "MULTILEVEL_TABU_RESULTS.csv")},
    "200": {
        "tabu": ("run_tabu_200.yml", "tabu-200-summary", "summary_200.csv"),
        "ml": ("run_multilevel_200.yml", "multilevel-200-summary", "summary_200.csv")}}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized(row):
    return {key.lower(): value for key, value in row.items()}


def valid(row):
    try:
        if row.get("feasible", "").upper() != "YES":
            return False
        for key in ("makespan", "fitness", "drone_violation", "capacity_violation", "waiting_violation"):
            value = float(row.get(key, 0))
            if not math.isfinite(value) or (key.endswith("violation") and abs(value) > 1e-6):
                return False
        runtime = float(row.get("runtime_s", row.get("runtime", "nan")))
        return float(row["makespan"]) > 0 and math.isfinite(runtime) and runtime >= 0
    except (ValueError, TypeError, KeyError):
        return False


def runtime(row):
    return float(row.get("runtime_s", row.get("runtime")))


def summarize(rows, expected):
    by_run = defaultdict(list)
    for row in rows:
        by_run[str(row.get("run", ""))].append(row)
    accepted = [items[0] for run, items in by_run.items()
                if run in {str(i) for i in range(1, expected+1)} and len(items) == 1 and valid(items[0])]
    best = min(accepted, key=lambda r: (float(r["makespan"]), int(r["run"]))) if accepted else None
    result = {"Rows": len(rows), "Valid_runs": len(accepted), "Expected_runs": expected,
              "Best": float(best["makespan"]) if best else "",
              "Mean": statistics.mean(float(r["makespan"]) for r in accepted) if accepted else "",
              "Worst": max(float(r["makespan"]) for r in accepted) if accepted else "",
              "Best_run": best["run"] if best else "", "Best_seed": best.get("seed", "") if best else "",
              "Best_solution": best.get("solution", "") if best else "",
              "Best_runtime_s": runtime(best) if best else "",
              "Mean_runtime_valid_s": statistics.mean(runtime(r) for r in accepted) if accepted else "",
              "Statuses": json.dumps(dict(Counter(r.get("feasible", "UNKNOWN") for r in rows)), sort_keys=True),
              "Duplicate_runs": sum(len(items) > 1 for items in by_run.values()),
              "Unexpected_runs": sum(run not in {str(i) for i in range(1, expected+1)} for run in by_run)}
    return result


def compare(tabu_rows, ml_rows, group, output, sha="local", run_ids=None):
    sizes = (50, 100) if group == "50_100" else (200,)
    expected = 5 if group == "50_100" else 3
    names = {f"{size}.{ratio}.{replica}" for size in sizes for ratio in (10, 20, 30, 40) for replica in range(1, 5)}
    grouped = {"tabu": defaultdict(list), "ml": defaultdict(list)}
    for side, rows in (("tabu", tabu_rows), ("ml", ml_rows)):
        for raw in rows:
            row = normalized(raw)
            name = row.get("instance", "")
            if name not in names:
                raise ValueError(f"Unexpected instance for {group}: {name}")
            grouped[side][name].append(row)
    result = []
    for name in sorted(names, key=lambda n: tuple(map(int, n.split(".")))):
        left, right = grouped["tabu"][name], grouped["ml"][name]
        tabu, ml = summarize(left, expected), summarize(right, expected)
        matched = len(left) == len(right) == expected
        seeds = []
        for rows in (left, right):
            seeds.append({str(r.get("run")): r.get("seed", "") for r in rows})
        same_seeds = "UNKNOWN"
        if matched and all(seeds[0].values()) and all(seeds[1].values()):
            same_seeds = "YES" if seeds[0] == seeds[1] and len(seeds[0]) == expected else "NO"
        comparable = tabu["Best"] != "" and ml["Best"] != ""
        complete = all(s["Valid_runs"] == expected and s["Rows"] == expected for s in (tabu, ml))
        row = {"Instance": name, "Size": int(name.split(".")[0]),
               "Comparison_status": "COMPLETE" if complete else "PARTIAL" if comparable else "NOT_COMPARABLE",
               "Same_seed_set": same_seeds}
        row.update({"Tabu_" + k: v for k, v in tabu.items()})
        row.update({"ML_" + k: v for k, v in ml.items()})
        row["Gap_best_ML_vs_Tabu_pct"] = 100*(ml["Best"]-tabu["Best"])/tabu["Best"] if comparable else ""
        row["Gap_mean_ML_vs_Tabu_pct"] = 100*(ml["Mean"]-tabu["Mean"])/tabu["Mean"] if comparable else ""
        row["Gap_mean_time_ML_vs_Tabu_pct"] = 100*(ml["Mean_runtime_valid_s"]-tabu["Mean_runtime_valid_s"])/tabu["Mean_runtime_valid_s"] if comparable and tabu["Mean_runtime_valid_s"] > 0 else ""
        row["Winner_best"] = ("ML" if ml["Best"] < tabu["Best"] else "TABU" if ml["Best"] > tabu["Best"] else "TIE") if comparable else "NOT_COMPARABLE"
        row["Mean_runtime_ratio_ML_over_Tabu"] = ml["Mean_runtime_valid_s"]/tabu["Mean_runtime_valid_s"] if comparable and tabu["Mean_runtime_valid_s"] > 0 else ""
        row["Source_sha"] = sha
        for side, run_id in (run_ids or {}).items():
            row[side + "_workflow_run_id"] = run_id
        result.append(row)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"COMPARE_ML_vs_TABU_{group}.csv"
    simple = []
    for row in result:
        short = {"Instance": row["Instance"],
                 "Tabu_Best": row["Tabu_Best"], "ML_Best": row["ML_Best"],
                 "Gap_Best_pct": row["Gap_best_ML_vs_Tabu_pct"],
                 "Tabu_Mean": row["Tabu_Mean"], "ML_Mean": row["ML_Mean"],
                 "Gap_Mean_pct": row["Gap_mean_ML_vs_Tabu_pct"],
                 "Tabu_Mean_time_s": row["Tabu_Mean_runtime_valid_s"],
                 "ML_Mean_time_s": row["ML_Mean_runtime_valid_s"],
                 "Gap_Mean_time_pct": row["Gap_mean_time_ML_vs_Tabu_pct"],
                 "Tabu_Valid_runs": row["Tabu_Valid_runs"], "ML_Valid_runs": row["ML_Valid_runs"],
                 "Status": row["Comparison_status"], "Winner": row["Winner_best"]}
        simple.append({k: round(v, 4) if isinstance(v, float) else v for k, v in short.items()})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(simple[0]))
        writer.writeheader(); writer.writerows(simple)
    # Keep provenance and diagnostics outside the short comparison table.
    (out / f"comparison_details_{group}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    complete = [r for r in result if r["Comparison_status"] == "COMPLETE"]
    counts = Counter(r["Winner_best"] for r in complete)
    text = (f"ML vs Tabu {group}: {len(complete)}/{len(result)} instances have all {expected} feasible runs on both sides. "
            f"Among complete instances: ML wins {counts['ML']}, Tabu wins {counts['TABU']}, ties {counts['TIE']}.\n\n"
            "Gap (%) = 100 * (ML - Tabu) / Tabu; negative favors ML. Compare makespan, not penalized fitness. "
            "PARTIAL comparisons use different numbers of feasible runs; inspect counts before interpreting. "
            "Timeout/error/infeasible/duplicate runs are excluded from best and mean. "
            "Runtime averages cover feasible runs only. Gap_Mean_time_pct compares the two mean runtimes; "
            "negative means ML is faster. Full diagnostics and best solutions (when available) are in comparison_details JSON. "
            "Same commit does not establish equal actual iterations/time limits.\n")
    (out / f"summary_{group}.md").write_text(text, encoding="utf-8")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)
    return result


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def api(url, token, binary=False):
    req = Request(url, headers={"Authorization": "Bearer " + token,
                               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    with build_opener(SafeRedirect()).open(req, timeout=30) as response:
        data = response.read()
    return data if binary else json.loads(data)


def fetch(group, side, repo, sha, branch, token, source_sha=None):
    workflow, artifact_name, filename = SOURCES[group][side]
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/") + f"/repos/{repo}/actions"
    query = urlencode({"head_sha": sha, "branch": branch, "per_page": 30})
    runs = api(f"{base}/workflows/{workflow}/runs?{query}", token)["workflow_runs"]
    for run in runs:
        if run["head_sha"] != sha:
            continue
        artifacts = api(f"{base}/runs/{run['id']}/artifacts?per_page=100", token)["artifacts"]
        for artifact in artifacts:
            if artifact["name"] != artifact_name or artifact["expired"]:
                continue
            try:
                data = api(f"{base}/artifacts/{artifact['id']}/zip", token, binary=True)
            except HTTPError as exc:
                if exc.code in (404, 410):
                    continue
                raise
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [n for n in archive.namelist() if n.split("/")[-1] == filename]
                if len(members) != 1:
                    continue
                # Read only the named CSV; do not extract or execute artifact contents.
                rows = list(csv.DictReader(io.StringIO(archive.read(members[0]).decode("utf-8-sig"))))
            expected_source = source_sha or sha
            if any(normalized(row).get("source_sha", expected_source) != expected_source for row in rows):
                continue
            if expected_source != sha and any("source_sha" not in normalized(row) for row in rows):
                continue
            print(f"Found {side} {group} at workflow run {run['id']}, commit {sha}")
            return rows, run["id"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=SOURCES, required=True)
    parser.add_argument("--tabu", help="Local Tabu CSV")
    parser.add_argument("--ml", help="Local Multilevel CSV")
    parser.add_argument("--output", default="comparison")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "local"))
    args = parser.parse_args()
    if args.remote:
        repo, branch, token = (os.environ.get(k) for k in ("GITHUB_REPOSITORY", "GITHUB_REF_NAME", "GH_TOKEN"))
        if not all((repo, branch, token)):
            parser.error("Remote comparison requires GITHUB_REPOSITORY, GITHUB_REF_NAME, GH_TOKEN")
        query_sha = args.sha
        if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
            # PR ref names and merge SHAs differ from the workflow-runs API head fields.
            base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
            current = api(f"{base}/repos/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}", token)
            query_sha, branch = current["head_sha"], current["head_branch"]
        found = {}
        for attempt in range(3):
            for side in ("tabu", "ml"):
                if side not in found:
                    item = fetch(args.group, side, repo, query_sha, branch, token, source_sha=args.sha)
                    if item:
                        found[side] = item
            if len(found) == 2:
                compare(found["tabu"][0], found["ml"][0], args.group, args.output, args.sha,
                        {s: found[s][1] for s in found})
                return
            if attempt < 2:
                time.sleep(5)
        message = ("Comparison pending: the other workflow has not uploaded its summary for this commit. "
                   "The workflow finishing later will compare. You can also compare downloaded CSVs locally.")
        print(message)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(message + "\n")
    else:
        if not args.tabu or not args.ml:
            parser.error("Provide both --tabu and --ml")
        compare(read_rows(args.tabu), read_rows(args.ml), args.group, args.output, args.sha)


if __name__ == "__main__":
    main()
