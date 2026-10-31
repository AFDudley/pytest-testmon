"""Dependency tracking beyond the Python lines a test executes in its own process.

testmon's core fingerprinting sees only the lines coverage.py traces in the process
that runs a test. A test can also depend on:

- a data file it opens for reading,
- code a Python child process executes,
- a data file a Python child process opens for reading,
- a file named on the command line of a non-Python child process.

This module gives each of those a fingerprint using the storage testmon already has
for Python source. A whole-file content block (see `testmon.process_code.Module`,
the `ext != "py"` branch of its `blocks` property) stands in for a data file's
fingerprint: `testmon_core.TestmonData.get_tests_fingerprints` already fingerprints
any file this way, Python or not, once the file's relative path and a covered-line
marker reach it through the normal `nodes_files_lines` structure.

A child process that cannot report its dependencies marks its test "always run".
`ALWAYS_RUN_SENTINEL` names a file that never exists on disk, so `TestmonData`
always treats it as changed and the test selector picks the test on every run. A
missed dependency this way never hides a real change.
"""

import json
import os
import random
import re
import sys
from functools import lru_cache
from subprocess import CalledProcessError, run

# A dependency filename that never exists on disk. A test with this dependency is
# always selected; see the module docstring and
# `TestmonData.get_tests_fingerprints`.
ALWAYS_RUN_SENTINEL = ".testmon-always-run"

# The sidecar file a Python child process writes for the open-file reads it recorded.
# See `write_sidecar`/`read_sidecars` and `SITECUSTOMIZE_SOURCE`.
SIDECAR_PREFIX = "testmon-subprocess-deps-"

_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".tox",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".tmontmp",
    }
)

_CODE_SUFFIXES = (".py", ".pyc")

_PYTHON_BASENAME_RE = re.compile(r"python\d*(\.\d+)?(\.exe)?$", re.IGNORECASE)


def _is_excluded(relpath):
    if relpath.startswith("../") or relpath in ("..", "."):
        return True
    parts = relpath.split("/")
    return any(part.startswith(".") or part in _EXCLUDED_DIR_NAMES for part in parts)


@lru_cache()
def _git_ls_files(abs_rootdir):
    try:
        result = run(
            ["git", "ls-files"],
            cwd=abs_rootdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (FileNotFoundError, NotADirectoryError, CalledProcessError):
        return None
    return frozenset(result.stdout.splitlines())


def get_git_tracked_files(rootdir):
    """All relpaths git tracks in `rootdir`'s repository, or `None` if `rootdir`
    is not inside a git repository (or `git` is unavailable).

    `None` disables the git-tracked restriction in `is_trackable_data_path` and
    `is_trackable_argv_path` rather than treating every data-file dependency as
    untracked: a project with no git repository has no tracked-file source of
    truth to defer to, so it keeps the old, path-shape-only rule.

    Cached on the resolved absolute path, never the raw `rootdir` string: an
    empty `rootdir` means "the current directory" (`TestmonData.for_local_run`
    passes it this way throughout the test suite), and the current directory
    changes between callers. Caching on the literal empty string would pin the
    very first caller's result for every later one.
    """
    return _git_ls_files(os.path.abspath(rootdir))


def is_trackable_data_path(relpath, rootdir):
    """True if a change to `relpath` under the project root should select a test.

    `relpath` is already relative to the project root, with forward slashes. The
    check excludes Python source (testmon already tracks it through coverage), VCS
    and cache directories, and any dotted path segment. Excluding dotted segments
    also excludes testmon's own `.testmondata` file and `.tmontmp` scratch
    directory, so a test never depends on testmon's own bookkeeping.

    A change to the file only selects a test when the file is tracked by git in
    `rootdir` -- the same source of truth testmon's Python-file fingerprinting
    already trusts (`process_code.get_source_sha` reads git's own blob sha for a
    tracked, unmodified file). A scratch or temp file a test writes inside the
    project directory, and never commits, is regenerated test run to test run
    with no real source change behind it; counting it as a dependency reselects
    the test on every run even when nothing in the project changed.
    """
    if _is_excluded(relpath):
        return False
    if relpath.endswith(_CODE_SUFFIXES):
        return False
    tracked = get_git_tracked_files(rootdir)
    if tracked is None:
        return True
    return relpath in tracked


def is_trackable_argv_path(relpath, rootdir):
    """True if a change to `relpath` named on a child's command line should select a test.

    Unlike `is_trackable_data_path`, a `.py` file named on a command line is not a
    file the parent process executed, so it stays in: the parent process has no
    other way to know the child depends on it. The git-tracked restriction
    (see `is_trackable_data_path`) still applies: an untracked file named on a
    command line is as unreliable a dependency signal as one read directly.
    """
    if _is_excluded(relpath):
        return False
    tracked = get_git_tracked_files(rootdir)
    if tracked is None:
        return True
    return relpath in tracked


def worker_scoped_tmontmp_dir(rootdir):
    """Absolute path to the current process's own `.tmontmp` scratch directory.

    Every `TestmonCollector` in one xdist session shares a project `rootdir`,
    but the sidecar directory must not be shared: `read_sidecars` deletes
    every sidecar file it finds in the directory it is given, not only the
    ones its own tests expect. Two xdist workers flushing at the same real
    time race for the same sidecar files. Confirmed by direct
    instrumentation (see `tests/test_dependency_channels.py::
    test_python_subprocess_xdist_batch_race_does_not_reselect_test`): one
    worker's flush reads, and deletes, a sidecar a DIFFERENT worker's child
    wrote for a test that worker has not flushed yet. The true owner then
    never sees its own child's report -- the file is already gone -- and
    marks the test always-run, even though the child reported cleanly and
    nothing on disk changed.

    `PYTEST_XDIST_WORKER` names the current xdist worker (`gw0`, `gw1`, ...).
    `xdist.remote` sets it in the worker process's own environment before
    pytest even starts, so it is stable for the worker's whole session. A
    `.tmontmp/<workerid>` subdirectory per worker gives every worker's
    children and flushes an isolated directory: no two processes ever list,
    read, or delete from the same path, so the race cannot happen -- by
    construction, not by timing luck. A local (non-xdist) run has no
    `PYTEST_XDIST_WORKER` and keeps the plain `.tmontmp` directory, same as
    before this function existed.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    parts = (".tmontmp", worker_id) if worker_id else (".tmontmp",)
    return os.path.abspath(os.path.join(rootdir, *parts))


def project_relpath(abspath, rootdir):
    """Return `abspath` relative to `rootdir` with forward slashes, or None if outside it."""
    try:
        relpath = os.path.relpath(abspath, rootdir)
    except ValueError:
        return None
    relpath = relpath.replace(os.sep, "/")
    if relpath.startswith("../") or relpath == "..":
        return None
    return relpath


def subprocess_child_touches_rootdir(cwd, argv, rootdir):
    """False only when there is positive evidence that a child process's
    working directory and every argument stay outside `rootdir`.

    A child with that evidence cannot depend on project source, Python or not,
    so `TestmonCollector.record_subprocess` records no dependency for it
    instead of marking its test always-run (`ALWAYS_RUN_SENTINEL`) -- the
    common shape of a test fixture that clones a git template, or runs a
    pinned interpreter, entirely under a scratch directory outside the
    project. A command with no path-shaped evidence either way (a bare
    `echo hello`, started with no explicit `cwd`) stays conservative and
    counts as touching the project: testmon genuinely does not know what it
    read.

    `argv` already excludes the executable itself; see `record_subprocess`.
    """
    explicit_cwd = cwd if isinstance(cwd, (str, bytes, os.PathLike)) else None
    outside_evidence = False
    if explicit_cwd is not None:
        abscwd = os.path.abspath(os.fspath(explicit_cwd))
        if os.path.isdir(abscwd):
            if project_relpath(abscwd, rootdir) is not None:
                return True
            outside_evidence = True
    for arg in argv:
        if not os.path.isabs(arg) or not os.path.exists(arg):
            continue
        if project_relpath(arg, rootdir) is not None:
            return True
        outside_evidence = True
    return not outside_evidence


def is_read_open(mode, flags):
    """True if a `sys.audit("open", file, mode, flags)` event opened the file for reading.

    `mode` is the string `open()` was called with, or None for `os.open()`-style
    calls, in which case `flags` carries the access mode bits instead.
    """
    if mode:
        return "r" in mode or "+" in mode
    if flags is None:
        return True
    return (flags & os.O_ACCMODE) != os.O_WRONLY


def looks_like_python_executable(executable, argv0):
    """True if a spawned process is a Python interpreter testmon can instrument.

    Checks both the resolved `executable` `subprocess.Popen` reports and the
    program name the caller asked for (`argv0`): `executable` is None on some
    platforms until the child process resolves it.
    """
    for candidate in (executable, argv0):
        if not candidate:
            continue
        name = os.path.basename(str(candidate))
        if _PYTHON_BASENAME_RE.fullmatch(name):
            return True
        try:
            if os.path.samefile(str(candidate), sys.executable):
                return True
        except OSError:
            pass
    return False


def write_sidecar(directory, context, files):
    """Write one child process's opened-file report for `context` (the test name).

    Called once, at child process exit. `read_sidecars` in the parent process reads
    and deletes the file.
    """
    os.makedirs(directory, exist_ok=True)
    name = f"{SIDECAR_PREFIX}{os.getpid()}-{random.randint(0, 1_000_000_000)}.json"
    path = os.path.join(directory, name)
    payload = {"context": context, "files": sorted(files)}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)


def read_sidecars(directory):
    """Read and delete every sidecar file in `directory`.

    Returns a list of `(context, files)` pairs, one per child process that reported.
    Called once per batch flush, in the parent process.
    """
    results = []
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return results
    for name in names:
        if not (name.startswith(SIDECAR_PREFIX) and name.endswith(".json")):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        results.append((payload.get("context"), payload.get("files", [])))
    return results


# Installed as `sitecustomize.py` in the scratch directory testmon prepends to a
# Python child's `PYTHONPATH`. `site` imports a module named `sitecustomize` at the
# start of every run that is not started with `-S`; because our directory comes
# first on `PYTHONPATH`, our copy is the one `site` finds. It chains to any other
# `sitecustomize` later on `sys.path`, so it does not shadow a project's own one.
#
# The child's own executed code is not recorded here: that reuses coverage's
# existing subprocess support (`COVERAGE_PROCESS_START`, read by the `.pth` file the
# `coverage` package installs), configured with a static `context` naming the test,
# so the lines the child executes join the parent test's coverage data once
# `TestmonCollector` combines the child's data file back in.
SITECUSTOMIZE_SOURCE = '''\
"""Installed by pytest-testmon for the duration of one test session.

Records every file this Python process opens for reading under the project root, so
the test that started it can depend on that file too. See
testmon.subprocess_support.
"""
import os
import sys


def _chain_other_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    search_path = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    try:
        import importlib.util

        for finder in sys.meta_path:
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec("sitecustomize", search_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return
    except Exception:  # pylint: disable=broad-except
        pass


_chain_other_sitecustomize()

_ROOT = os.environ.get("TESTMON_SUBPROCESS_ROOT")
_DEPS_DIR = os.environ.get("TESTMON_SUBPROCESS_DIR")

if _ROOT and _DEPS_DIR:
    try:
        import atexit

        from testmon.subprocess_support import (
            is_read_open,
            is_trackable_data_path,
            project_relpath,
            write_sidecar,
        )

        _opened = set()

        def _on_open(path, mode, flags):
            if not isinstance(path, (str, bytes, os.PathLike)):
                return
            if not is_read_open(mode, flags):
                return
            relpath = project_relpath(os.path.abspath(os.fspath(path)), _ROOT)
            if relpath is None or not is_trackable_data_path(relpath, _ROOT):
                return
            _opened.add(relpath)

        def _audit(event, args):
            if event == "open":
                _on_open(*args)

        sys.addaudithook(_audit)

        def _flush():
            context = os.environ.get("TESTMON_SUBPROCESS_CONTEXT", "")
            write_sidecar(_DEPS_DIR, context, _opened)

        atexit.register(_flush)
    except Exception:  # pylint: disable=broad-except
        pass
'''


def install_sitecustomize(directory):
    """Write `SITECUSTOMIZE_SOURCE` into `directory`. Idempotent; call once per session."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "sitecustomize.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(SITECUSTOMIZE_SOURCE)
    return path


def prepend_pythonpath(directory):
    """Put `directory` first on `PYTHONPATH`. Returns the prior value, for restoring it."""
    previous = os.environ.get("PYTHONPATH")
    existing = previous or ""
    parts = [p for p in existing.split(os.pathsep) if p and p != directory]
    os.environ["PYTHONPATH"] = os.pathsep.join([directory] + parts)
    return previous


def restore_pythonpath(previous):
    """Undo `prepend_pythonpath`."""
    if previous is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = previous
