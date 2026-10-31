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
- test_binary_data_file_dependency_fingerprints_by_raw_bytes: a data file
  with binary (non-UTF-8-decodable) content. testmon.process_code fingerprints
  it from its raw bytes, never through tokenize/source_encoding.
- test_untracked_scratch_file_dependency_does_not_reselect_test: a data file
  inside the project directory but never committed to git -- the shape of a
  scratch/temp file a test regenerates with new content on every run
  (testmon.subprocess_support.is_trackable_data_path's git-tracked rule).
- test_subprocess_entirely_outside_rootdir_does_not_force_rerun: a child
  process whose cwd and every argument stay outside the project directory
  (testmon.subprocess_support.subprocess_child_touches_rootdir).

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


def test_tried_directory_open_is_not_tracked_as_a_data_file_dependency(testdir):
    """`open(a_directory)` fires the same `sys.audit("open", ...)` event as a
    normal file read, BEFORE Python's own `IsADirectoryError` check runs — so
    code that tries to open a path and catches that error (instead of
    checking `os.path.isdir` first) still leaves the attempt recorded. A
    directory has no byte content to fingerprint; recording one crashed
    `read_source_sha` with `IsADirectoryError` once that test's coverage was
    saved and reported back to the xdist controller
    (`TestmonCollector.record_opened_file`, `pytest_runtest_logreport`'s
    worker branch)."""
    testdir.makepyfile(
        test_a="""
        def test_tries_to_open_a_directory():
            try:
                open("a_directory")
            except IsADirectoryError:
                pass
        """
    )
    testdir.mkdir("a_directory")
    result = testdir.runpytest_subprocess("--testmon", "-n", "1")
    assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()
    result.assert_outcomes(passed=1)


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


def test_binary_data_file_dependency_fingerprints_by_raw_bytes(testdir):
    """A gzip (binary) data file dependency never crashes testmon, and still
    selects/deselects the test that reads it.

    Measured 2026-10-31: a test that reads a non-Python file whose bytes are
    not valid UTF-8 (e.g. a .gz archive) crashed every run with
    `read_source_sha crashed ... SyntaxError('invalid or missing encoding
    declaration')` then `UnicodeDecodeError`, because the fingerprint went
    through `coverage.phystokens.source_encoding` -> `tokenize.detect_encoding`,
    built for Python source. The fix fingerprints a non-Python file from its
    raw bytes directly, skipping that decode.
    """
    import gzip

    testdir.makepyfile(
        test_a="""
        import gzip

        def test_reads_gzip():
            with gzip.open("data.gz", "rb") as fh:
                fh.read()
        """
    )
    gz_path = str(testdir.tmpdir.join("data.gz"))
    with gzip.open(gz_path, "wb") as fh:
        fh.write(b"v1")

    result = testdir.runpytest_subprocess("--testmon")
    assert result.ret == 0, result.outlines
    result.assert_outcomes(passed=1)

    # Nothing on disk changed: the second run must not crash (the regression
    # was an INTERNALERROR here), and the test must settle stable, the same
    # check `test_data_file_dependency_selects_test` above uses.
    result = testdir.runpytest_subprocess("--testmon")
    assert result.ret == 0, result.outlines
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_reads_gzip" not in td.unstable_test_names

    # A change to the gzip bytes reselects the test.
    with gzip.open(gz_path, "wb") as fh:
        fh.write(b"v2")
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_reads_gzip" in td.unstable_test_names


def test_untracked_scratch_file_dependency_does_not_reselect_test(testdir):
    """A scratch file inside the project directory, never added to git -- the
    shape of a temp probe, a log, or a generated cache entry a test
    regenerates with new content on every run. `is_trackable_data_path` only
    counts a data-file dependency when git tracks the file in the project's
    rootdir, so an untracked scratch file must not reselect the test that
    reads it on an otherwise unchanged second run. Measured 2026-10-31,
    against the exophial test suite: with no git check, 861 of its tests
    carried a dependency on files shaped like this one and reselected on
    every run, changed or not.
    """
    testdir.makepyfile(
        test_a="""
        import pathlib

        def test_reads_scratch_file():
            pathlib.Path("scratch_probe.txt").read_text()
        """
    )
    testdir.tmpdir.join("scratch_probe.txt").write("v1")
    testdir.run("git", "init", "-q", "-b", "main")
    testdir.run("git", "config", "user.email", "t@t")
    testdir.run("git", "config", "user.name", "t")
    testdir.run("git", "add", "test_a.py")
    testdir.run("git", "commit", "-q", "-m", "seed")
    testdir.runpytest_subprocess("--testmon")

    # The scratch file was never committed. A real scratch file is rewritten
    # with different content on every run with no project edit behind it;
    # simulate that here.
    testdir.tmpdir.join("scratch_probe.txt").write("v2, a different run, same test code")
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_reads_scratch_file" not in td.unstable_test_names


def test_subprocess_entirely_outside_rootdir_does_not_force_rerun(testdir):
    """A child process whose cwd and every argument stay outside the project
    directory cannot depend on project source. Unlike
    test_untraceable_subprocess_always_reruns_test above (a child that
    touches the project directory but names no file testmon can track), it
    must not force a rerun on an otherwise unchanged run -- the shape of a
    test fixture that clones a git template, or runs a pinned interpreter,
    entirely under its own scratch directory outside the project.
    """
    testdir.makepyfile(
        test_a="""
        import subprocess, tempfile

        def test_spawns_outside_project():
            with tempfile.TemporaryDirectory() as outside:
                subprocess.run(["true"], cwd=outside, check=True)

        def test_plain():
            assert 1 == 1
        """
    )
    testdir.runpytest_subprocess("--testmon")

    # Nothing on disk changed. The subprocess never touched the project
    # directory, so it carries no possible project-source dependency.
    with CoreTestmonData.for_local_run("") as td:
        td.determine_stable()
        assert "test_a.py::test_spawns_outside_project" not in td.unstable_test_names
        assert "test_a.py::test_plain" not in td.unstable_test_names
