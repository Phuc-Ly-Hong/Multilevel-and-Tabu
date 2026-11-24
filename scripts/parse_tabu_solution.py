"""Extract the last complete final solution printed by the no_lwt solvers."""
import json
from pathlib import Path
import re


def parse_solution(content):
    if "TIMEOUT" in content or "ERROR_EXIT_CODE" in content:
        return {"solution": "", "solution_status": "INTERRUPTED"}
    current = None
    makespan_seen = False
    final = None
    for line in content.splitlines():
        if line.strip() == "Route details:":
            current = {}
            makespan_seen = False
            final = None
            continue
        if current is None:
            continue
        match = re.fullmatch(r"Vehicle\s+(\d+):\s*([\d\s-]*)", line.strip())
        if match:
            vehicle = int(match[1])
            if vehicle in current:
                current = None
                continue
            current[vehicle] = [int(node) for node in match[2].split()]
        elif line.startswith("Makespan:"):
            makespan_seen = True
        elif line.startswith("Fitness:"):
            if makespan_seen and current and sorted(current) == list(range(len(current))):
                final = [current[v] for v in range(len(current))]
            current = None
    if final is None:
        return {"solution": "", "solution_status": "MISSING"}
    # PARSED means extracted, not a new feasibility certificate; Feasible is separate.
    return {"solution": json.dumps(final, separators=(",", ":")), "solution_status": "PARSED"}


def read_solution(path):
    try:
        return parse_solution(Path(path).read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return {"solution": "", "solution_status": "NO_OUTPUT"}
