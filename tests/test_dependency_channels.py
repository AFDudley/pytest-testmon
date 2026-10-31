"""Acceptance tests for the dependency channels beyond Python line coverage.

See testmon/subprocess_support.py for the design. Each test covers one channel:

- test_data_file_dependency_selects_test: a data file the test itself opens for
  reading (testmon.subprocess_support.is_trackable_data_path).
- test_python_subprocess_data_file_dependency: a data file a Python child process
  opens for reading (the sidecar report in SITECUSTOMIZE_SOURCE).
- test_non_python_subprocess_argv_file_dependency: a file named on a non-Python
  child's command line (TestmonCollector.record_subprocess).
- test_untraceable_subprocess_always_reruns_test: a child whose dependencies
  cannot be recorded (ALWAYS_RUN_SENTINEL).

Every test runs `--testmon` once to build the dependency database, changes
something on disk (or nothing, for the always-run case), then calls
`determine_stable` directly to check which tests the next run would select,
the same pattern `tests/test_testmon.py` uses throughout.
"""

from testmon.testmon_core import TestmonData as CoreTestmonData

pytest_plugins = ("pytester",)


def test_data_file_dependency_selects_test(testdir):
    testdir.makepyfile(
        test_a="""
        import pathlib

        def test_reads_data():
            pathlib.Path("data.txt").read_text()

        def test_reads_other():
            pathlib.Path("other.txt").read_text()
        """
    )
    testdir.tmpdir.join("data.txt").write("v1")
    testdir.tmpdir.join("other.txt").write("o1")
    testdir.runpytest_subprocess("--testmon")

    testdir.tmpdir.join("other.txt").write("o2")
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_reads_other" in td.unstable_test_names
        assert "test_a.py::test_reads_data" not in td.unstable_test_names


def test_python_subprocess_data_file_dependency(testdir):
    testdir.makepyfile(
        test_a="""
        import subprocess, sys

        def test_child_reads_data():
            subprocess.run(
                [sys.executable, "-c", "open('child_data.txt').read()"], check=True
            )

        def test_child_reads_other():
            subprocess.run(
                [sys.executable, "-c", "open('child_other.txt').read()"], check=True
            )
        """
    )
    testdir.tmpdir.join("child_data.txt").write("v1")
    testdir.tmpdir.join("child_other.txt").write("o1")
    testdir.runpytest_subprocess("--testmon")

    testdir.tmpdir.join("child_other.txt").write("o2")
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_child_reads_other" in td.unstable_test_names
        assert "test_a.py::test_child_reads_data" not in td.unstable_test_names


def test_non_python_subprocess_argv_file_dependency(testdir):
    testdir.makepyfile(
        test_a="""
        import subprocess

        def test_cat_reads_data():
            subprocess.run(
                ["cat", "cat_data.txt"], check=True, stdout=subprocess.DEVNULL
            )

        def test_cat_reads_other():
            subprocess.run(
                ["cat", "cat_other.txt"], check=True, stdout=subprocess.DEVNULL
            )
        """
    )
    testdir.tmpdir.join("cat_data.txt").write("v1")
    testdir.tmpdir.join("cat_other.txt").write("o1")
    testdir.runpytest_subprocess("--testmon")

    testdir.tmpdir.join("cat_other.txt").write("o2")
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_cat_reads_other" in td.unstable_test_names
        assert "test_a.py::test_cat_reads_data" not in td.unstable_test_names


def test_untraceable_subprocess_always_reruns_test(testdir):
    testdir.makepyfile(
        test_a="""
        import subprocess

        def test_untraceable():
            subprocess.run(["echo", "hello"], check=True, stdout=subprocess.DEVNULL)

        def test_plain():
            assert 1 == 1
        """
    )
    testdir.runpytest_subprocess("--testmon")

    # Nothing on disk changed. An ordinary test is now stable; the test that
    # started an untraceable child is still selected on every run.
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_untraceable" in td.unstable_test_names
        assert "test_a.py::test_plain" not in td.unstable_test_names
