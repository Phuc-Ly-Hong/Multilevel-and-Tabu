"""Check actual tabu insertions in both solvers on a small original instance."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]


class SharedTenureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = ROOT / "tests" / ("tenure_audit_" + uuid.uuid4().hex)
        cls.work.mkdir()
        def build(name):
            source = (ROOT / "src" / (name + ".cpp")).read_text(encoding="utf-8-sig")
            marker = "            best_move.tenure = TABU_TENURE;"
            assert source.count(marker) == 1
            source = source.replace(marker, marker + '''
            cout << "AUDIT_TENURE " << ORIGINAL_CUSTOMERS << ' ' << num_nodes-1
                 << ' ' << TABU_TENURE << endl;''')
            marker = "            tabu_list.push_back(reverse_move);"
            source = source.replace(marker, '''            if (reverse_move.tenure != best_move.tenure)
                throw runtime_error("Forward/reverse tenure differs");
''' + marker)
            path = cls.work / (name + ".cpp")
            path.write_text(source, encoding="utf-8")
            result = subprocess.run(["g++", "-O1", "-std=c++17", "-I", str(ROOT / "src"),
                                     "-o", str(cls.work / (name + ".exe")), str(path)],
                                    capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise AssertionError(result.stderr)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(build, ("Tabu_no_lwt", "Multilevel_Tabu_no_lwt")))
        except Exception:
            cls.clean()
            raise

    @classmethod
    def clean(cls):
        path = cls.work.resolve()
        assert path.is_relative_to((ROOT / "tests").resolve())
        shutil.rmtree(path)

    @classmethod
    def tearDownClass(cls):
        cls.clean()

    def execute(self, name, *args):
        result = subprocess.run([str(self.work / (name + ".exe")), str(ROOT / "instances/10.10.1.txt"),
                                 "--seed", "4321", "--iter_k", "1", *args],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [tuple(map(int, m)) for m in re.findall(r"AUDIT_TENURE (\d+) (\d+) (\d+)", result.stdout)]

    def test_each_accepted_move_is_sampled_and_reproducible(self):
        for name in ("Tabu_no_lwt", "Multilevel_Tabu_no_lwt"):
            with self.subTest(name=name):
                rows = self.execute(name)
                self.assertTrue(rows)
                self.assertTrue(all(n == 10 and tenure in (1, 2) for n, active, tenure in rows))
                self.assertEqual({r[2] for r in rows}, {1, 2})
                self.assertEqual(rows, self.execute(name))

    def test_multilevel_bounds_never_use_reduced_customer_count(self):
        rows = self.execute("Multilevel_Tabu_no_lwt")
        reduced = [r for r in rows if r[1] < r[0]]
        self.assertTrue(reduced)
        # Reduced levels can still receive tenure 2, based on the original ten customers.
        self.assertTrue(any(tenure == 2 for n, active, tenure in reduced))

    def test_equal_bounds_are_exact_in_both_solvers(self):
        for name in ("Tabu_no_lwt", "Multilevel_Tabu_no_lwt"):
            rows = self.execute(name, "--tabu1", "0.3", "--tabu2", "0.3")
            self.assertTrue(rows)
            self.assertTrue(all(r[2] == 3 for r in rows))


if __name__ == "__main__":
    unittest.main()
