import os
import subprocess
import sys
import unittest


class TestCLI(unittest.TestCase):
    def test_module_execution_prints_hello(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(["src", env.get("PYTHONPATH", "")])
        result = subprocess.run(
            [sys.executable, "-m", "benchkitv2.cli"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "Hello, world!\n")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
