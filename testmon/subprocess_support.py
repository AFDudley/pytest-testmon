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


def is_trackable_data_path(relpath):
    """True if a change to `relpath` under the project root should select a test.

    `relpath` is already relative to the project root, with forward slashes. The
    check excludes Python source (testmon already tracks it through coverage), VCS
    and cache directories, and any dotted path segment. Excluding dotted segments
    also excludes testmon's own `.testmondata` file and `.tmontmp` scratch
    directory, so a test never depends on testmon's own bookkeeping.
    """
    if _is_excluded(relpath):
        return False
    return not relpath.endswith(_CODE_SUFFIXES)


def is_trackable_argv_path(relpath):
    """True if a change to `relpath` named on a child's command line should select a test.

    Unlike `is_trackable_data_path`, a `.py` file named on a command line is not a
    file the parent process executed, so it stays in: the parent process has no
    other way to know the child depends on it.
    """
    return not _is_excluded(relpath)


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
            if relpath is None or not is_trackable_data_path(relpath):
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
