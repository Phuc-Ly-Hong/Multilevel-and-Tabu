import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_multilevel_tabu as compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.output = ROOT / "tests" / ("compare_fixture_" + uuid.uuid4().hex)
        self.output.mkdir()
        self.quiet = patch("sys.stdout", new=io.StringIO())
        self.quiet.start()

    def tearDown(self):
        self.quiet.stop()
        target = self.output.resolve()
        assert target.is_relative_to((ROOT / "tests").resolve())
        shutil.rmtree(target)

    def rows(self, upper=True, size=50, cost=100):
        result = []
        for run in range(1, 6 if size != 200 else 4):
            row = {"instance": f"{size}.10.1", "run": str(run), "seed": str(run),
                   "makespan": str(cost+run-1), "fitness": str(cost+run-1),
                   "runtime": "2", "feasible": "YES", "drone_violation": "0",
                   "capacity_violation": "0", "waiting_violation": "0"}
            if upper:
                row = {k.title(): v for k, v in row.items()}
                row["Runtime_s"] = row.pop("Runtime")
            result.append(row)
        return result

    def test_best_mean_gap_and_both_schemas(self):
        rows = compare.compare(self.rows(cost=100), self.rows(False, cost=90), "50_100", self.output)
        r = rows[0]
        self.assertEqual(len(rows), 32)
        self.assertEqual((r["Tabu_Best"], r["ML_Best"], r["Tabu_Mean"], r["ML_Mean"]), (100,90,102,92))
        self.assertEqual(r["Gap_best_ML_vs_Tabu_pct"], -10)
        self.assertEqual(r["Winner_best"], "ML")
        self.assertEqual(r["Same_seed_set"], "YES")
        self.assertEqual(r["Comparison_status"], "COMPLETE")

    def test_200_three_runs_and_missing_instances(self):
        rows = compare.compare(self.rows(False,200), self.rows(False,200,95), "200", self.output)
        self.assertEqual(len(rows),16)
        self.assertEqual(rows[0]["ML_Expected_runs"],3)
        self.assertEqual(rows[-1]["Comparison_status"],"NOT_COMPARABLE")
        self.assertEqual(rows[-1]["Gap_best_ML_vs_Tabu_pct"],"")

    def test_timeout_infeasible_nonfinite_and_zero_not_best(self):
        ml = self.rows(False)
        for row in ml[:4]: row["makespan"]="0"
        ml[0]["feasible"]="TIMEOUT"
        ml[1]["feasible"]="NO"
        ml[2]["makespan"]="nan"
        ml[3]["makespan"]="inf"
        r=compare.compare(self.rows(),ml,"50_100",self.output)[0]
        self.assertEqual(r["ML_Valid_runs"],1)
        self.assertEqual(r["ML_Best"],104)
        self.assertEqual(r["Comparison_status"],"PARTIAL")

    def test_duplicate_seed_and_violation_reporting(self):
        ml=self.rows(False)
        ml.append(dict(ml[0]))
        ml[1]["capacity_violation"]="1"
        r=compare.compare(self.rows(),ml,"50_100",self.output)[0]
        self.assertEqual(r["ML_Duplicate_runs"],1)
        self.assertEqual(r["ML_Valid_runs"],3)
        self.assertEqual(r["ML_Best"],102)

    def test_historical_csv_seeds_unknown(self):
        ml=self.rows(False)
        for row in ml: row.pop("seed")
        r=compare.compare(self.rows(),ml,"50_100",self.output)[0]
        self.assertEqual(r["Same_seed_set"],"UNKNOWN")

    def test_remote_uses_exact_commit_and_named_csv(self):
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,"w") as z:
            text=io.StringIO();rows=self.rows(False)
            writer=csv.DictWriter(text,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            z.writestr("TABU_RESULTS.csv",text.getvalue())
        calls=[]
        def api(url,token,binary=False):
            calls.append(url)
            if "/workflows/" in url:
                return {"workflow_runs":[{"id":1,"head_sha":"wrong"},{"id":2,"head_sha":"same"}]}
            if "/runs/2/" in url:
                return {"artifacts":[{"name":"tabu-results","expired":False,"id":3}]}
            if url.endswith("/artifacts/3/zip"):
                self.assertTrue(binary)
                return archive.getvalue()
            raise AssertionError(url)
        with patch.object(compare,"api",side_effect=api):
            rows,run=compare.fetch("50_100","tabu","owner/repo","same","Method_B","TOKEN")
        self.assertEqual(run,2)
        self.assertEqual(len(rows),5)
        self.assertFalse(any("/runs/1/" in url for url in calls))

    def test_workflow_parsers_export_status_and_seed(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is needed only for workflow integration checks")
        previous=Path.cwd()
        try:
            for name in ("run_tabu.yml","run_multilevel_tabu.yml","run_tabu_200.yml","run_multilevel_200.yml"):
                folder=self.output/name;folder.mkdir();os.chdir(folder)
                data=yaml.load((ROOT/".github/workflows"/name).read_text(encoding="utf-8"),Loader=yaml.BaseLoader)
                blocks=[step["run"].split("python3 <<'EOF'",1)[1].rsplit("EOF",1)[0]
                        for job in data["jobs"].values() for step in job.get("steps",[])
                        if "python3 <<'EOF'" in step.get("run","")]
                for run in range(1,6):
                    content="Makespan: 10\nDrone violation: 0\nFitness: 10\n"
                    if run==2:content+="TIMEOUT\n"
                    if run==3:content+="ERROR_EXIT_CODE_1\n"
                    Path(f"output_run{run}.txt").write_text(content)
                    Path(f"runtime_run{run}.txt").write_text("1.0")
                instance="200.10.1" if "_200" in name else "50.10.1"
                for block in blocks[:-1]:
                    exec(compile(block.replace("${{ matrix.instance }}",instance),name,"exec"),{})
                    result=json.loads(Path("results.json").read_text())
                    self.assertEqual(result["runs"][0]["seed"],1,name)
                    self.assertEqual(result["runs"][1]["feasible"],"TIMEOUT",name)
                    self.assertEqual(result["runs"][2]["feasible"],"ERROR",name)
                artifact="results/part" if "_200" in name else ("multilevel-result-" if "multilevel" in name else "tabu-result-")+instance
                target=Path(artifact);target.mkdir(parents=True)
                shutil.copyfile("results.json",target/"results.json")
                exec(compile(blocks[-1],name,"exec"),{})
                path=next(Path(".").glob("*.csv"))
                row=compare.read_rows(path)[0]
                self.assertEqual(row.get("Seed",row.get("seed")),"1",name)
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
