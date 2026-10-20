"""Independent CVRP solution checker; does not call the ML executable.

Public solution:
  python scripts/check_neo_solution.py instance.vrp --sol instance.sol --edges
ML results (all rows or one instance):
  python scripts/check_neo_solution.py --csv results/NEO_CVRP.csv
  python scripts/check_neo_solution.py --csv results/NEO_CVRP.csv --instance ulysses-n22-k4 --edges
Direct routes (VRP IDs, depot included):
  python scripts/check_neo_solution.py --instance NAME --routes "[[1,2,3,1],...]"

NEO .sol customer index + 1 = original VRP node ID.
The ML CSV stores VRP node IDs with depot at both ends.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read_vrp(path):
    """Read demands/depot and rebuild the declared distance matrix independently."""
    path = Path(path)
    headers, sections, section = {}, {}, None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper() == "EOF":
            break
        if line.endswith("_SECTION"):
            section = sections.setdefault(line, [])
        elif section is not None:
            section.extend(line.split())
        else:
            key, value = line.split(":", 1)
            headers[key.strip().upper()] = value.strip()
    if headers.get("TYPE") != "CVRP":
        raise ValueError("Expected TYPE CVRP")
    n, capacity = int(headers["DIMENSION"]), float(headers["CAPACITY"])
    depots = [int(v) for v in sections["DEPOT_SECTION"] if int(v) != -1]
    if len(depots) != 1:
        raise ValueError("Expected exactly one depot")
    depot = depots[0]
    tokens = sections["DEMAND_SECTION"]
    if len(tokens) != 2 * n:
        raise ValueError("Missing or extra demand rows")
    demand = {int(tokens[i]): float(tokens[i + 1]) for i in range(0, len(tokens), 2)}
    if set(demand) != set(range(1, n + 1)) or demand.get(depot) != 0:
        raise ValueError("Invalid node IDs or depot demand")
    if capacity <= 0 or any(not math.isfinite(v) or v < 0 or v > capacity for v in demand.values()):
        raise ValueError("Invalid capacity/demand")
    match = re.search(r"-k(\d+)", path.stem, re.I)
    if not match:
        match = re.search(r"(?:No of trucks|Min no of trucks)\s*:\s*(\d+)", headers.get("COMMENT", ""), re.I)
    vehicles = int(headers["VEHICLES"]) if "VEHICLES" in headers else int(match[1]) if match else None
    if vehicles is None or vehicles <= 0:
        raise ValueError("Cannot infer fleet size")
    metric = headers["EDGE_WEIGHT_TYPE"]
    distance = [[0] * n for _ in range(n)]
    if metric == "EXPLICIT":
        values = iter(map(int, sections["EDGE_WEIGHT_SECTION"]))
        fmt = headers["EDGE_WEIGHT_FORMAT"]
        if fmt == "FULL_MATRIX":
            pairs = ((i, j) for i in range(n) for j in range(n))
        elif fmt == "LOWER_DIAG_ROW":
            pairs = ((i, j) for i in range(n) for j in range(i + 1))
        elif fmt == "LOWER_ROW":
            pairs = ((i, j) for i in range(n) for j in range(i))
        elif fmt == "UPPER_ROW":
            pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
        else:
            raise ValueError("Unsupported EDGE_WEIGHT_FORMAT: " + fmt)
        try:
            for i, j in pairs:
                weight = next(values)
                if weight < 0:
                    raise ValueError("Negative edge weight")
                distance[i][j] = weight
                if fmt != "FULL_MATRIX":
                    distance[j][i] = weight
        except StopIteration as exc:
            raise ValueError("Truncated edge weights") from exc
        if next(values, None) is not None:
            raise ValueError("Extra edge weights")
    else:
        if metric not in ("EUC_2D", "GEO"):
            raise ValueError("Unsupported metric: " + metric)
        tokens = sections["NODE_COORD_SECTION"]
        if len(tokens) != 3 * n:
            raise ValueError("Missing or extra coordinates")
        coords = {int(tokens[i]): (float(tokens[i + 1]), float(tokens[i + 2]))
                  for i in range(0, len(tokens), 3)}
        if set(coords) != set(demand) or any(not math.isfinite(v) for xy in coords.values() for v in xy):
            raise ValueError("Invalid coordinate IDs/values")

        def geo_radians(value):
            # TSPLIB GEO is DDD.MM (degrees/minutes), not decimal degrees.
            # For example, 38.30 means 38 degrees 30 minutes = 38.5 degrees.
            degrees = int(value)
            minutes = value - degrees
            return math.radians(degrees + 5 * minutes / 3)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = coords[i + 1], coords[j + 1]
                if metric == "GEO":
                    lat1, lon1 = map(geo_radians, a)
                    lat2, lon2 = map(geo_radians, b)
                    q1 = math.cos(lon1 - lon2)
                    q2 = math.cos(lat1 - lat2)
                    q3 = math.cos(lat1 + lat2)
                    cosine = ((1 + q1) * q2 - (1 - q1) * q3) / 2
                    weight = int(6378.388 * math.acos(max(-1, min(1, cosine))) + 1)
                else:
                    # Round EACH EDGE, not the sum of a route. Do not use
                    # Python round(), whose half-to-even rule differs.
                    weight = math.floor(math.dist(a, b) + 0.5)
                distance[i][j] = distance[j][i] = weight
    if any(distance[i][i] != 0 for i in range(n)):
        raise ValueError("Nonzero matrix diagonal")
    return {"instance": path.stem, "metric": metric, "dimension": n,
            "capacity": capacity, "vehicles": vehicles, "depot": depot,
            "demand": demand, "distance": distance}


def check_solution(vrp_path, routes, reported_cost=None, *, numbering="vrp", has_depots=True):
    """Check CVRP feasibility and recompute total distance, without running ML.

    routes: list of integer routes.
    numbering='vrp', has_depots=True: original node IDs including end depots.
    numbering='neo', has_depots=False: NEO .sol customer indices only.
    A malformed closed route is rejected, never silently repaired.
    Returns a report with each route's load/cost and every edge's distance.
    """
    p = read_vrp(vrp_path)
    if numbering not in ("vrp", "neo"):
        raise ValueError("numbering must be vrp or neo")
    if reported_cost is not None:
        reported_cost = float(reported_cost)
        if not math.isfinite(reported_cost) or reported_cost < 0:
            raise ValueError("Invalid reported cost")
    if not isinstance(routes, list):
        raise ValueError("routes must be a list")
    seen, details, errors, total, used = Counter(), [], [], 0, 0
    cost_complete = True
    for index, raw in enumerate(routes, 1):
        if not isinstance(raw, list) or any(type(node) is not int for node in raw):
            errors.append(f"Route {index}: expected a list of integer node IDs")
            cost_complete = False
            continue
        tour = [node + 1 for node in raw] if numbering == "neo" else list(raw)
        if not has_depots:
            tour = [p["depot"]] + tour + [p["depot"]]
        if len(tour) < 2 or tour[0] != p["depot"] or tour[-1] != p["depot"]:
            errors.append(f"Route {index}: must start and end at depot {p['depot']}")
            cost_complete = False
            continue
        if any(node not in p["demand"] for node in tour):
            errors.append(f"Route {index}: unknown node ID")
            cost_complete = False
            continue
        customers = tour[1:-1]
        if p["depot"] in customers:
            errors.append(f"Route {index}: interior depot (multi-trip is not allowed)")
        if customers:
            used += 1
        seen.update(node for node in customers if node != p["depot"])
        load = sum(p["demand"][node] for node in customers)
        edges = [{"from": a, "to": b, "distance": p["distance"][a - 1][b - 1]}
                 for a, b in zip(tour, tour[1:])]
        cost = sum(edge["distance"] for edge in edges)
        total += cost
        details.append({"route": index, "nodes": tour, "load": load,
                        "capacity": p["capacity"], "cost": cost, "edges": edges})
        if load > p["capacity"] + 1e-9:
            errors.append(f"Route {index}: load {load:g} exceeds capacity {p['capacity']:g}")
    missing = [node for node in p["demand"] if node != p["depot"] and seen[node] == 0]
    duplicates = sorted(node for node, count in seen.items() if count > 1)
    if missing:
        errors.append("Missing customers: " + str(missing))
    if duplicates:
        errors.append("Repeated customers: " + str(duplicates))
    if used > p["vehicles"]:
        errors.append(f"Used {used} vehicles; allowed {p['vehicles']}")
    matches = None if reported_cost is None or not cost_complete else abs(total - reported_cost) < 1e-6
    status = ("INVALID_SOLUTION" if errors else "FEASIBLE_NO_REPORTED_COST"
              if matches is None else "PASS" if matches else "COST_MISMATCH")
    return {"instance": p["instance"], "metric": p["metric"], "status": status,
            "feasible": not errors, "vehicles_used": used, "vehicles_limit": p["vehicles"],
            "computed_cost": total, "cost_complete": cost_complete, "reported_cost": reported_cost,
            "difference": None if reported_cost is None or not cost_complete else total - reported_cost,
            "cost_matches": matches, "missing": missing, "duplicates": duplicates,
            "errors": errors, "routes": details}


def read_sol(path):
    """Parse alternatives separately; never concatenate Solution 1 and Solution 2."""
    groups = []
    current = {"label": "Solution", "routes": [], "cost": None}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"solution\b", line, re.I):
            if current["routes"]:
                groups.append(current)
            current = {"label": line.rstrip(":"), "routes": [], "cost": None}
        elif re.match(r"r?oute\s*#?\d+\s*:", line, re.I):
            current["routes"].append([int(v) for v in line.split(":", 1)[1].split()])
        elif re.match(r"cost\b", line, re.I):
            current["cost"] = float(line[4:].strip(" :\t"))
        else:
            raise ValueError("Unrecognized .sol line: " + line)
    if current["routes"]:
        groups.append(current)
    if not groups:
        raise ValueError("No routes found in .sol")
    return groups


def check_routes(instance, routes, reported_cost=None, *, dataset_root=None,
                 numbering="vrp", has_depots=True):
    """Check routes by dataset name (e.g. ulysses-n22-k4) or a .vrp path.

    Returns a dict: feasible, computed_cost, errors, routes, etc.
    Default input uses original VRP node IDs, with depot at both ends.
    """
    path = Path(instance)
    if not path.is_file():
        root = Path(dataset_root) if dataset_root is not None else ROOT / "NEO_CVRP_62_pairs"
        matches = [p for p in root.rglob("*.vrp") if p.stem.casefold() == path.stem.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Expected one dataset named {instance}; found {len(matches)}")
        path = matches[0]
    return check_solution(path, routes, reported_cost, numbering=numbering, has_depots=has_depots)


def print_report(report, *, detailed=False, edges=False, label=""):
    claimed = "not provided" if report["reported_cost"] is None else f"{report['reported_cost']:g}"
    computed = report["computed_cost"] if report["cost_complete"] else "NOT_COMPUTABLE"
    print(f"{report['instance']} {label}: {report['status']} metric={report['metric']} "
          f"computed={computed} reported={claimed} feasible={report['feasible']}")
    if detailed or edges:
        print(f"  Vehicles: {report['vehicles_used']}/{report['vehicles_limit']}; IDs: original VRP node IDs")
        for route in report["routes"]:
            print(f"  Route {route['route']}: {' -> '.join(map(str, route['nodes']))}")
            print(f"    load={route['load']:g}/{route['capacity']:g}, cost={route['cost']}")
            if edges:
                for edge in route["edges"]:
                    print(f"    {edge['from']} -> {edge['to']}: {edge['distance']}")
        total_label = "Total" if report["cost_complete"] else "Partial sum (some routes could not be evaluated)"
        print(f"  {total_label} = " + " + ".join(str(r["cost"]) for r in report["routes"]) + f" = {report['computed_cost']}")
        if report["difference"] is not None:
            print(f"  Computed minus reported = {report['difference']:g}")
    for error in report["errors"]:
        print("  ERROR: " + error)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vrp", nargs="?", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sol", type=Path)
    group.add_argument("--csv", type=Path)
    group.add_argument("--routes", help="JSON list of routes supplied directly")
    parser.add_argument("--instance", help="Dataset name for --routes, or CSV instance filter")
    parser.add_argument("--cost", type=float, help="Optional reported cost to check with --routes")
    parser.add_argument("--customers-only", action="store_true", help="Direct routes omit end depots")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "NEO_CVRP_62_pairs")
    parser.add_argument("--solution-column")
    parser.add_argument("--cost-column")
    parser.add_argument("--numbering", choices=("neo", "vrp"))
    parser.add_argument("--edges", action="store_true", help="Show every traversed edge")
    args = parser.parse_args()
    reports = []
    try:
        if args.routes is not None:
            instance = args.vrp or args.instance
            if instance is None:
                parser.error("Supply --instance NAME or a .vrp path with --routes")
            numbering = args.numbering or "vrp"
            report = check_routes(instance, json.loads(args.routes), args.cost,
                                  dataset_root=args.dataset_root, numbering=numbering,
                                  has_depots=not args.customers_only and numbering != "neo")
            print("VALID = " + ("YES" if report["feasible"] else "NO"))
            objective = report["computed_cost"] if report["cost_complete"] else "NOT_COMPUTABLE"
            print(f"OBJECTIVE_TOTAL_DISTANCE = {objective}")
            print_report(report, detailed=True, edges=args.edges)
            reports.append(report)
        elif args.sol:
            if not args.vrp:
                parser.error("Supply the .vrp path before --sol")
            for solution in read_sol(args.sol):
                report = check_solution(args.vrp, solution["routes"], solution["cost"],
                                        numbering=args.numbering or "neo", has_depots=False)
                print_report(report, detailed=True, edges=args.edges, label=solution["label"])
                reports.append(report)
        else:
            with args.csv.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            instances = {p.stem: p for p in args.dataset_root.rglob("*.vrp")}
            for row in rows:
                name = row["instance"]
                if args.instance and name != args.instance:
                    continue
                if args.vrp and name != args.vrp.stem:
                    continue
                column = args.solution_column or ("solution" if "solution" in row else "my_solution_vrp_node_ids")
                cost_column = args.cost_column or ("cost" if "cost" in row else "my_cost")
                report = check_solution(args.vrp or instances[name], json.loads(row[column]), float(row[cost_column]),
                                        numbering=args.numbering or "vrp", has_depots=args.numbering != "neo")
                print_report(report, detailed=bool(args.instance or args.vrp), edges=args.edges,
                             label=f"run={row.get('run', '?')} seed={row.get('seed', '?')}")
                reports.append(report)
            if not reports:
                raise ValueError("No CSV rows matched")
        print("SUMMARY " + " ".join(f"{key}={value}" for key, value in sorted(Counter(r["status"] for r in reports).items())))
        return 2 if any(not r["feasible"] or r["cost_matches"] is False for r in reports) else 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print("ERROR: " + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
