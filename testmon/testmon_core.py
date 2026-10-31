import glob
import hashlib
import os
import random
import sys
import sysconfig
import textwrap
from functools import lru_cache
from collections import defaultdict
from xmlrpc.client import Fault, ProtocolError, ServerProxy
from socket import gaierror
from typing import TypeVar

try:
    from pytest_cov.plugin import CovPlugin
except ImportError:
    pass
import pytest
from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

from testmon import db
from testmon import TESTMON_VERSION as TM_CLIENT_VERSION
from testmon.common import (
    get_logger,
    get_system_packages,
    drop_patch_version,
    git_current_head,
)

from testmon.subprocess_support import (
    ALWAYS_RUN_SENTINEL,
    get_git_tracked_files,
    install_sitecustomize,
    is_read_open,
    is_trackable_argv_path,
    is_trackable_data_path,
    looks_like_python_executable,
    prepend_pythonpath,
    project_relpath,
    read_sidecars,
    restore_pythonpath,
    subprocess_child_touches_rootdir,
)

from testmon.process_code import (
    match_fingerprint,
    create_fingerprint,
    methods_to_checksums,
    get_files_shas,
    get_source_sha,
    Module,
)

from testmon.common import DepsNOutcomes, TestExecutions

T = TypeVar("T")

TEST_BATCH_SIZE = 250

CHECKUMS_ARRAY_TYPE = "I"
DB_FILENAME = ".testmondata"

logger = get_logger(__name__)


def get_data_file_path():
    return os.environ.get("TESTMON_DATAFILE", DB_FILENAME)


def home_file(test_execution_name):
    return test_execution_name.split("::", 1)[0]


def is_python_file(file_path):
    return file_path[-3:] == ".py"


class TestmonException(Exception):
    pass


class SourceTree:
    """
    - reads files from file_system and caches them in memory
      (if the file was once read, let's use the on memory
      copy for further operations. Also caches the mtime.)
    - store rootdir and convert between relative and absolute paths
    - implement the check if the file is changed based on mtime, fsha.
    - implement the ckeck if the node is stable based on fingerprint
    - mockability (unit tests without filesystem, check if the file is changed
      based on mtime, fsha)
    """

    def __init__(self, rootdir, packages=None):
        self.rootdir = rootdir
        self.packages = packages
        self.cache: dict = {}

    def get_file(self, filename):
        if filename not in self.cache:
            is_python = filename.endswith(".py")
            code, fsha = get_source_sha(
                directory=self.rootdir,
                filename=filename,
                is_python=is_python,
            )
            if fsha:
                try:
                    fs_mtime = os.path.getmtime(os.path.join(self.rootdir, filename))
                    self.cache[filename] = Module(
                        source_code=code,
                        mtime=fs_mtime,
                        # A data-file dependency (testmon.subprocess_support) can
                        # name a file with no extension (e.g. "LICENSE"), unlike
                        # the .py files this split previously always saw. "py"
                        # vs. non-"py" is the only distinction `Module` makes, so
                        # extensionless names fold into the non-"py" whole-file
                        # block path instead of raising on the missing dot.
                        ext="py" if is_python else filename.rsplit(".", 1)[-1],
                        fs_fsha=fsha,
                        filename=filename,
                        rootdir=self.rootdir,
                    )
                except FileNotFoundError:
                    self.cache[filename] = None
            else:
                self.cache[filename] = None
        return self.cache[filename]


def check_mtime(file_system: SourceTree, record):
    absfilename = os.path.join(file_system.rootdir, record["filename"])

    cache_module = file_system.cache.get(record["filename"], None)
    try:
        fs_mtime = cache_module.mtime if cache_module else os.path.getmtime(absfilename)
    except OSError:
        return False
    return record["mtime"] == fs_mtime


def check_fsha(file_system, record):
    cache_module = file_system.get_file(record["filename"])
    fs_fsha = cache_module.fs_fsha if cache_module else None

    return record["fsha"] == fs_fsha


def check_fingerprint(
    disk, record: db.ChangedFileData
):  # filename name method_fshas id failed
    file = record[0]
    fingerprint = record[2]

    module = disk.get_file(file)
    return module and match_fingerprint(module, fingerprint)


def split_filter(disk, function, records: [T]) -> ([T], [T]):
    first = []
    second = []
    for record in records:
        if function(disk, record):
            first.append(record)
        else:
            second.append(record)
    return first, second


@lru_cache(maxsize=1000)
def should_include(cov, filename):
    return cov._should_trace(str(filename), None).trace


def collect_mhashes(source_tree, new_changed_file_data):
    files_mhashes = {}
    for filename in new_changed_file_data:
        module = source_tree.get_file(filename)
        files_mhashes[filename] = module.method_checksums if module else None
    return files_mhashes


class TestmonData:  # pylint: disable=too-many-instance-attributes
    __test__ = False

    def __init__(
        self,
        rootdir,
        database=None,
        readonly=False,
    ):
        self.rootdir = rootdir
        self.source_tree = SourceTree(rootdir=self.rootdir)

        if database:
            self.db = database  # pylint: disable=invalid-name
        else:
            self.db = db.DB(
                os.path.join(self.rootdir, get_data_file_path()), readonly=readonly
            )  # pylint: disable=invalid-name

        # Initialize instance variables (will be set by init methods)
        self.environment = None
        self.exec_id = None
        self.system_packages_change = None
        self.files_of_interest = None
        self.all_files = {}
        self.unstable_test_names = None
        self.unstable_files = None
        self.stable_test_names = None
        self.stable_files = None
        self.failing_tests = None

    @classmethod
    def for_local_run(
        cls,
        rootdir,
        database=None,
        environment=None,
        system_packages=None,
        python_version=None,
    ):
        instance = cls(rootdir, database=database, readonly=False)
        instance._init_for_local_run(environment, system_packages, python_version)
        return instance

    def _init_for_local_run(
        self,
        environment=None,
        system_packages=None,
        python_version=None,
    ):
        """Internal method to initialize for local run."""
        self.environment = environment if environment else "default"

        if system_packages is None:
            system_packages = get_system_packages()
        system_packages = drop_patch_version(system_packages)

        if not python_version:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Initiate execution (controller or single process)
        try:
            result = self.db.initiate_execution(
                self.environment,
                system_packages,
                python_version,
                {
                    "tm_client_version": TM_CLIENT_VERSION,
                    "git_head_sha": git_current_head(),
                    "ci": os.environ.get("CI"),
                },
            )
        except (ConnectionRefusedError, Fault, ProtocolError, gaierror) as exc:
            logger.error(
                (
                    "%s error when communication with testmon.net. (falling back to"
                    " .testmondata locally)"
                ),
                exc,
            )
            self.db = db.DB(os.path.join(self.rootdir, get_data_file_path()))  # pylint: disable=invalid-name
            result = self.db.initiate_execution(
                self.environment, system_packages, python_version, {}
            )

        self.exec_id = result["exec_id"]
        self.system_packages_change = result["packages_changed"]
        self.files_of_interest = result["filenames"]

    @classmethod
    def for_worker(  # pylint: disable=too-many-arguments
        cls,
        rootdir,
        exec_id,
        database=None,
        system_packages_change=None,
        files_of_interest=None,
        environment=None,
    ):
        instance = cls(rootdir, database=database, readonly=True)
        instance._init_for_worker(
            exec_id, system_packages_change, files_of_interest, environment
        )
        return instance

    def _init_for_worker(
        self,
        exec_id,
        system_packages_change=None,
        files_of_interest=None,
        environment=None,
    ):
        """Internal method to initialize for worker run."""
        self.exec_id = exec_id
        self.system_packages_change = system_packages_change
        self.files_of_interest = files_of_interest
        if environment:
            self.environment = environment
        elif not hasattr(self, "environment") or self.environment is None:
            self.environment = "default"

    @property
    def new_db(self):
        # Only local DB instances have file_created attribute
        if isinstance(self.db, db.DB):
            return self.db.file_created
        return False

    def close_connection(self):
        """Release the storage: the sqlite connection, or the transport of a tmnet proxy.

        Idempotent. Tests that inspect the data after a run reopen it (see tests/lib4tests.py).
        """
        database, self.db = self.db, None
        if database is None:
            return
        if isinstance(database, ServerProxy):
            database("close")()  # attribute access would be a remote call
        else:
            database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close_connection()

    @property
    def all_tests(self):
        return self.db.all_test_executions(self.exec_id)

    def get_tests_fingerprints(self, nodes_files_lines, reports) -> TestExecutions:
        test_executions_fingerprints = {}
        for context in nodes_files_lines:
            deps_n_outcomes: DepsNOutcomes = {"deps": []}

            for filename, covered in nodes_files_lines[context].items():
                if filename == ALWAYS_RUN_SENTINEL:
                    # Never on disk: `TestmonData.determine_stable` always treats
                    # it as changed, so the test it is attached to is always
                    # selected. See `TestmonCollector._merge_extra_dependencies`.
                    deps_n_outcomes["deps"].append(
                        {
                            "filename": ALWAYS_RUN_SENTINEL,
                            "mtime": None,
                            "fsha": "always-run",
                            "method_checksums": [],
                        }
                    )
                elif os.path.exists(os.path.join(self.rootdir, filename)):
                    module = self.source_tree.get_file(filename)
                    fingerprint = create_fingerprint(module, covered)
                    deps_n_outcomes["deps"].append(
                        {
                            "filename": filename,
                            "mtime": module.mtime,
                            "fsha": module.fs_fsha,
                            "method_checksums": fingerprint,
                        }
                    )

            deps_n_outcomes.update(process_result(reports[context]))
            deps_n_outcomes["forced"] = context in self.stable_test_names and (
                context not in self.failing_tests
            )
            test_executions_fingerprints[context] = deps_n_outcomes
        return test_executions_fingerprints

    def sync_db_fs_tests(self, retain):
        collected = retain.union(set(self.stable_test_names))
        add = list(collected - set(self.all_tests))
        test_execution_file_fps = {
            test_name: {
                "deps": (
                    {
                        "filename": home_file(test_name),
                        "method_checksums": methods_to_checksums(["0match"]),
                        "mtime": None,
                        "fsha": None,
                    },
                )
            }
            for test_name in add
            if is_python_file(home_file(test_name))
        }
        if test_execution_file_fps:
            self.save_test_execution_file_fps(test_execution_file_fps)

        to_delete = list(set(self.all_tests) - collected)
        self.db.delete_test_executions(to_delete, self.exec_id)

    def determine_stable(self):
        files_fshas = {}
        for filename in self.files_of_interest:
            module = self.source_tree.get_file(filename)
            if module:
                files_fshas[filename] = module.fs_fsha

        # Compare the fshas from disk to the fshas in the database and get files
        # where the fsha is not in database.
        new_changed_file_data = self.db.fetch_unknown_files(files_fshas, self.exec_id)

        # Get the mhashes for the files from above
        files_mhashes = collect_mhashes(self.source_tree, new_changed_file_data)

        tests = self.db.determine_tests(self.exec_id, files_mhashes)
        affected_tests, self.failing_tests = tests["affected"], tests["failing"]

        self.all_files = set(self.db.filenames(self.exec_id))
        self.unstable_test_names = set()
        self.unstable_files = set()

        for fingerprint_miss in affected_tests:
            self.unstable_test_names.add(fingerprint_miss)
            self.unstable_files.add(fingerprint_miss.split("::", 1)[0])

        self.stable_test_names = set(self.all_tests) - self.unstable_test_names
        self.stable_files = set(self.all_files) - self.unstable_files

    @property
    def avg_durations(self) -> dict:
        stats = defaultdict(lambda: {"test_execution": 0, "sum_duration": 0})

        for (
            test_execution_id,
            report,
        ) in self.all_tests.items():  # pylint: disable=no-member
            if report:
                class_name = get_test_execution_class_name(test_execution_id)
                module_name = get_test_execution_module_name(test_execution_id)

                stats[test_execution_id]["test_execution"] += 1
                stats[test_execution_id]["sum_duration"] = report.get("duration") or 0
                if class_name:
                    stats[class_name]["test_execution"] += 1
                    stats[class_name]["sum_duration"] += stats[test_execution_id][
                        "sum_duration"
                    ]
                stats[module_name]["test_execution"] += 1
                stats[module_name]["sum_duration"] += stats[test_execution_id][
                    "sum_duration"
                ]

        durations = defaultdict(lambda: 0)
        for key, stats in stats.items():
            durations[key] = stats["sum_duration"] / stats["test_execution"]

        return durations

    def save_test_execution_file_fps(self, test_executions_fingerprints):
        self.db.insert_test_file_fps(test_executions_fingerprints, self.exec_id)

    def fetch_saving_stats(self, select):
        return self.db.fetch_saving_stats(self.exec_id, select)


def get_new_mtimes(filesystem, hits):
    """hits: [(filename, _, _, fingerprint_id)]"""
    try:
        for hit in hits:
            module = filesystem.get_file(hit[0])
            if module:
                yield module.mtime, module.fs_fsha, hit[3]
    except KeyError:
        for hit in hits:
            module = filesystem.get_file(hit["filename"])
            if module:
                yield module.mtime, module.fs_fsha, hit["fingerprint_id"]


def get_test_execution_class_name(node_id):
    if len(node_id.split("::")) > 2:
        return node_id.split("::")[1]
    return None


def get_test_execution_module_name(node_id):
    return node_id.split("::")[0]


@lru_cache(1000)
def cached_relpath(path, basepath):
    return os.path.relpath(path, basepath).replace(os.sep, "/")


# coverage 7.10+ defaults to the sys.monitoring core on CPython 3.14+. That core
# doesn't support dynamic contexts: sys.monitoring disables a line event once the
# line has been recorded, so a line executed by several tests of one batch is
# attributed to the first test only and the other tests lose the dependency.
# testmon relies on contexts_by_lineno, so it needs the C tracer.
COVERAGE_CORE = "ctrace"


def force_context_capable_core(cov: Coverage):
    try:
        cov.set_option("run:core", COVERAGE_CORE)
    except CoverageException:
        # coverage < 7.10 has no core option and never defaults to sys.monitoring.
        pass


class TestmonCollector:
    coverage_stack: [Coverage] = []
    cov_owners: dict = {}
    _audit_hook_installed = False

    def __init__(
        self, rootdir, testmon_labels=None, cov_plugin=None
    ):  # TODO remove cov_plugin
        try:
            from testmon.testmon_core import (  # pylint: disable=import-outside-toplevel
                Testmon as UberTestmon,
            )

            TestmonCollector.coverage_stack: [Coverage] = UberTestmon.coverage_stack
        except ImportError:
            pass
        if testmon_labels is None:
            testmon_labels = {"singleprocess"}
        self.rootdir = rootdir
        self.testmon_labels = testmon_labels
        self.cov: Coverage = None
        self.sub_cov_file = None
        self.tmontmp_dir = None
        self._subprocess_includes = None
        self._prev_pythonpath = None
        self.cov_plugin: CovPlugin = cov_plugin
        self._test_name = None
        self._next_test_name = None
        self.batched_test_names = set()
        self.check_stack = []
        self.is_started = False
        self._interrupted_at = None
        # Dependencies this process observed that coverage.py's line tracer does
        # not record: data files opened for reading (`record_opened_file`), and
        # files named on a non-Python child's command line (`record_subprocess`).
        # Keyed by test name, merged into `nodes_files_lines` at the next batch
        # flush; see `_merge_extra_dependencies`.
        self.file_deps: dict = {}
        # Tests whose dependencies this process could not fully record: a child
        # process that is not Python with no project file on its command line, or
        # a Python child whose report did not arrive by the next flush. See
        # `ALWAYS_RUN_SENTINEL`.
        self.always_run: set = set()
        # test name -> count of Python child processes `record_subprocess` saw
        # started for that test, so `_merge_extra_dependencies` can tell whether
        # every child reported back.
        self._expected_python_children: dict = {}
        # Resolve and cache the git-tracked-file set for `rootdir` now, before
        # any test runs and the subprocess audit hook is live. `is_trackable_data_path`
        # and `is_trackable_argv_path` call `get_git_tracked_files`, which shells
        # out to `git ls-files`. A cache miss triggered mid-test would audit
        # that very subprocess call as a child the running test spawned,
        # falsely marking the test always-run. A cache hit, warmed here before
        # `self._test_name` is ever set, never spawns the subprocess.
        #
        # `process_code.get_files_shas` shells out to `git ls-files --stage -m`
        # for the same reason (a fast path for Python-file fshas); warm it here
        # too, for the same reason.
        get_git_tracked_files(self.rootdir)
        get_files_shas(self.rootdir)
        TestmonCollector._ensure_audit_hook()

    @classmethod
    def _ensure_audit_hook(cls):
        if cls._audit_hook_installed:
            return
        sys.addaudithook(_testmon_audit_hook)
        cls._audit_hook_installed = True

    def record_opened_file(self, path, mode, flags):
        """Record a data file this process opened for reading, for the running test."""
        if self._test_name is None:
            return
        if not isinstance(path, (str, bytes, os.PathLike)):
            return  # a bare file descriptor carries no filename
        if not is_read_open(mode, flags):
            return
        abspath = os.path.abspath(os.fspath(path))
        # `sys.audit("open", ...)` also fires for a directory listing
        # (`os.scandir`, `os.listdir`, `Path.iterdir`). A directory has no byte
        # content to fingerprint, and recording one here crashes
        # `read_source_sha` later with `IsADirectoryError` when `determine_stable`
        # tries to open it as a file.
        if os.path.isdir(abspath):
            return
        relpath = project_relpath(abspath, self.rootdir)
        if relpath is None or not is_trackable_data_path(relpath, self.rootdir):
            return
        self.file_deps.setdefault(self._test_name, set()).add(relpath)

    def record_subprocess(self, executable, args, cwd):
        """Record a child process the running test started.

        A Python child's dependencies arrive later, through its own coverage data
        file and sidecar report; see `_merge_extra_dependencies`. A non-Python
        child's command line is inspected directly: each argument naming an
        existing, git-tracked project file becomes a dependency. A child whose
        working directory and arguments carry no evidence of touching the
        project directory (`subprocess_child_touches_rootdir`) cannot depend on
        project source; testmon records no dependency for it rather than
        marking its test always-run. A child that DOES touch the project, but
        names no trackable file, and a Python child whose report never
        arrives, mark the test always-run.
        """
        if self._test_name is None:
            return
        test_name = self._test_name
        argv = [
            os.fspath(arg)
            for arg in (args or [])
            if isinstance(arg, (str, bytes, os.PathLike))
        ]
        argv0 = argv[0] if argv else None
        touches_rootdir = subprocess_child_touches_rootdir(cwd, argv, self.rootdir)
        if looks_like_python_executable(executable, argv0):
            if touches_rootdir:
                self._expected_python_children[test_name] = (
                    self._expected_python_children.get(test_name, 0) + 1
                )
            return
        base_dir = cwd if isinstance(cwd, (str, bytes, os.PathLike)) else self.rootdir
        found_file = False
        for arg in argv:
            candidate = arg if os.path.isabs(arg) else os.path.join(base_dir, arg)
            if not os.path.isfile(candidate):
                continue
            relpath = project_relpath(os.path.abspath(candidate), self.rootdir)
            if relpath is None or not is_trackable_argv_path(relpath, self.rootdir):
                continue
            self.file_deps.setdefault(test_name, set()).add(relpath)
            found_file = True
        if not found_file and touches_rootdir:
            self.always_run.add(test_name)

    def start_cov(self):
        if not self.cov._started:
            TestmonCollector.coverage_stack.append(self.cov)
            self.cov.start()

    def stop_cov(self):
        if self.cov is None:
            return
        assert self.cov in TestmonCollector.coverage_stack
        if TestmonCollector.coverage_stack:
            while TestmonCollector.coverage_stack[-1] != self.cov:
                cov = TestmonCollector.coverage_stack.pop()
                cov.stop()
        if self.cov._started:
            self.cov.stop()
            TestmonCollector.coverage_stack.pop()
        if TestmonCollector.coverage_stack:
            TestmonCollector.coverage_stack[-1].start()

    def setup_coverage(self, subprocess=False):
        params = {
            "include": [os.path.join(self.rootdir, "*")],
            "omit": {
                os.path.join(value, "*")
                for key, value in sysconfig.get_paths().items()
                if key.endswith("lib")
            },
        }
        if self.cov_plugin and self.cov_plugin._started:
            cov = self.cov_plugin.cov_controller.cov
            TestmonCollector.coverage_stack.append(cov)
            if cov.config.source:
                params["include"] = list(
                    set(
                        [os.path.join(self.rootdir, "*")]
                        + [
                            os.path.join(os.path.abspath(source), "*")
                            for source in cov.config.source
                        ]
                    )
                )
            elif cov.config.run_include:
                params["include"] = list(
                    set(cov.config.run_include + params["include"])
                )
            # params["omit"] = cov.config.run_omit
            if cov.config.branch:
                raise TestmonException(
                    "testmon doesn't support simultaneous run with pytest-cov when "
                    "branch coverage is on. Please disable branch coverage."
                )
        if subprocess:
            self.setup_subprocess(params["include"])
        self.cov = Coverage(data_file=self.sub_cov_file, config_file=False, **params)
        self.cov._warn_no_data = False
        force_context_capable_core(self.cov)
        TestmonCollector.cov_owners[id(self.cov)] = self
        if TestmonCollector.coverage_stack:
            TestmonCollector.coverage_stack[-1].stop()

        self.start_cov()

    def start_testmon(self, test_name, next_test_name=None):
        self._next_test_name = next_test_name

        self.batched_test_names.add(test_name)
        if self.cov is None:
            self.setup_coverage(subprocess=True)

        self.start_cov()
        self._test_name = test_name
        self.cov.switch_context(test_name)
        if self.sub_cov_file:
            self._write_subprocess_rc(test_name)
            os.environ["TESTMON_SUBPROCESS_CONTEXT"] = test_name
        self.check_stack = TestmonCollector.coverage_stack.copy()

    def discard_current(self):
        self._interrupted_at = self._test_name

    def get_batch_coverage_data(self):
        if self.check_stack != TestmonCollector.coverage_stack:
            pytest.exit(
                (
                    "Exiting pytest!!!! This test corrupts Testmon.coverage_stack:"
                    f" {self._test_name} {self.check_stack},"
                    f" {TestmonCollector.coverage_stack}"
                ),
                returncode=3,
            )

        nodes_files_lines = {}

        if self.cov and (
            len(self.batched_test_names) >= TEST_BATCH_SIZE
            or self._next_test_name is None
            or self._interrupted_at
        ):
            self.cov.stop()
            self._combine_subprocess_coverage()
            nodes_files_lines, lines_data = self.get_nodes_files_lines(
                dont_include=self._interrupted_at
            )
            self._merge_extra_dependencies(nodes_files_lines)

            if (
                len(TestmonCollector.coverage_stack) > 1
                and TestmonCollector.coverage_stack[-1] == self.cov
            ):
                filtered_lines_data = {
                    file: data
                    for file, data in lines_data.items()
                    if should_include(TestmonCollector.coverage_stack[-2], file)
                }
                TestmonCollector.coverage_stack[-2].get_data().add_lines(
                    filtered_lines_data
                )

            self.cov.erase()
            self.cov.start()
            self.batched_test_names = set()
        return nodes_files_lines

    def _combine_subprocess_coverage(self):
        """Merge a Python child's own executed-code data into this batch's data.

        Relies on coverage's own subprocess support: a child started with
        `COVERAGE_PROCESS_START` set writes a parallel data file next to
        `self.sub_cov_file`, already carrying the running test as its context
        (see `_write_subprocess_rc`). `Coverage.combine` merges it in and
        deletes it.
        """
        if not self.sub_cov_file:
            return
        child_files = [
            path
            for path in glob.glob(self.sub_cov_file + ".*")
            if not path.endswith("_rc")
        ]
        if not child_files:
            return
        self.cov.combine(data_paths=child_files, strict=False, keep=False)

    def _merge_extra_dependencies(self, nodes_files_lines):
        """Fold data-file and subprocess dependencies into this batch's fingerprints.

        `nodes_files_lines` already has one entry per batched test (see
        `get_nodes_files_lines`); a data-file dependency is added to it the same
        way a covered Python line is: as a filename with an arbitrary covered-line
        marker, which `TestmonData.get_tests_fingerprints` fingerprints by whole
        file content for any non-Python file. A sidecar a Python child wrote (see
        `testmon.subprocess_support.SITECUSTOMIZE_SOURCE`) is read, deleted, and
        merged the same way, into the test named by its own `context` field. A
        test whose expected child reports did not all arrive is added to
        `always_run`.
        """
        arrived = {}
        if self.tmontmp_dir:
            for context, files in read_sidecars(self.tmontmp_dir):
                if not context:
                    continue
                arrived[context] = arrived.get(context, 0) + 1
                if files:
                    self.file_deps.setdefault(context, set()).update(files)

        for test_name, count in self._expected_python_children.items():
            if arrived.get(test_name, 0) < count:
                self.always_run.add(test_name)
        self._expected_python_children = {}

        for test_name, relpaths in self.file_deps.items():
            if test_name not in nodes_files_lines:
                continue
            for relpath in relpaths:
                nodes_files_lines[test_name].setdefault(relpath, {1})
        self.file_deps = {}

        for test_name in self.always_run:
            if test_name not in nodes_files_lines:
                continue
            nodes_files_lines[test_name][ALWAYS_RUN_SENTINEL] = {1}
        self.always_run = set()

    def get_nodes_files_lines(self, dont_include):
        cov_data: CoverageData = self.cov.get_data()
        files = cov_data.measured_files()
        nodes_files_lines = {}
        files_lines = {}
        for file in files:
            relfilename = cached_relpath(file, self.rootdir)

            contexts_by_lineno = cov_data.contexts_by_lineno(file)

            for lineno, contexts in contexts_by_lineno.items():
                for context in contexts:
                    nodes_files_lines.setdefault(context, {}).setdefault(
                        relfilename, set()
                    ).add(lineno)
                    files_lines.setdefault(file, set()).add(lineno)
        nodes_files_lines.pop(dont_include, None)
        self.batched_test_names.discard(dont_include)
        nodes_files_lines.pop("", None)
        for test_name in self.batched_test_names:
            if home_file(test_name) not in nodes_files_lines.setdefault(test_name, {}):
                nodes_files_lines[test_name].setdefault(home_file(test_name), {1})
        return nodes_files_lines, files_lines

    def close(self):
        if self.cov is None:
            return
        assert self.cov in TestmonCollector.coverage_stack
        if TestmonCollector.coverage_stack:
            while TestmonCollector.coverage_stack[-1] != self.cov:
                cov = TestmonCollector.coverage_stack.pop()
                cov.stop()
        if self.cov._started:
            self.cov.stop()
            TestmonCollector.coverage_stack.pop()
        TestmonCollector.cov_owners.pop(id(self.cov), None)
        if self.sub_cov_file:
            os.remove(self.sub_cov_file + "_rc")
            os.environ.pop("COVERAGE_CORE", None)
            os.environ.pop("TESTMON_SUBPROCESS_ROOT", None)
            os.environ.pop("TESTMON_SUBPROCESS_DIR", None)
            os.environ.pop("TESTMON_SUBPROCESS_CONTEXT", None)
            restore_pythonpath(self._prev_pythonpath)
        os.environ.pop("COVERAGE_PROCESS_START", None)
        self.cov = None
        if TestmonCollector.coverage_stack:
            TestmonCollector.coverage_stack[-1].start()

    def setup_subprocess(self, includes):
        # Gives a Python child process testmon's own file-open tracking (through a
        # `sitecustomize.py` on its `PYTHONPATH`) and its executed-code coverage
        # (through coverage's own `COVERAGE_PROCESS_START` subprocess support,
        # pinned to the running test's name as a static context). See
        # `testmon.subprocess_support`.
        self.tmontmp_dir = os.path.abspath(os.path.join(self.rootdir, ".tmontmp"))
        os.makedirs(self.tmontmp_dir, exist_ok=True)
        self.sub_cov_file = os.path.join(
            self.tmontmp_dir, ".testmoncoverage" + str(random.randint(0, 1000000))
        )
        self._subprocess_includes = includes
        self._write_subprocess_rc("")
        os.environ["COVERAGE_PROCESS_START"] = self.sub_cov_file + "_rc"
        os.environ["COVERAGE_CORE"] = COVERAGE_CORE
        os.environ["TESTMON_SUBPROCESS_ROOT"] = self.rootdir
        os.environ["TESTMON_SUBPROCESS_DIR"] = self.tmontmp_dir
        install_sitecustomize(self.tmontmp_dir)
        self._prev_pythonpath = prepend_pythonpath(self.tmontmp_dir)

    def _write_subprocess_rc(self, context):
        rc_content = textwrap.dedent(
            """\
                [run]
                data_file = {}
                include = {}
                parallel=True
                context = {}
                """
        ).format(
            self.sub_cov_file,
            "\n ".join(self._subprocess_includes),
            context,
        )
        with open(self.sub_cov_file + "_rc", "w", encoding="utf-8") as subprocess_rc:
            subprocess_rc.write(rc_content)


def _active_collector():
    if not TestmonCollector.coverage_stack:
        return None
    cov = TestmonCollector.coverage_stack[-1]
    return TestmonCollector.cov_owners.get(id(cov))


_audit_hook_warned = False


def _testmon_audit_hook(event, args):
    """Route one `sys.audit` event to the collector running the current test.

    `sys.audit` fires this for every `open()` and `subprocess.Popen()` call in the
    process, so a bug here must never raise: that would turn an unrelated, correct
    `open()` call elsewhere in the process into a failure. A problem is logged once
    and the event is dropped; the dropped test still has `determine_stable`'s
    existing "no record runs" rule and, for a missed child-process dependency, the
    always-run fallback in `TestmonCollector.record_subprocess`.
    """
    global _audit_hook_warned
    try:
        collector = _active_collector()
        if collector is None:
            return
        if event == "open":
            collector.record_opened_file(*args)
        elif event == "subprocess.Popen":
            executable, popen_args, cwd, _env = args
            collector.record_subprocess(executable, popen_args, cwd)
    except Exception:  # pylint: disable=broad-except
        if not _audit_hook_warned:
            _audit_hook_warned = True
            logger.warning(
                "testmon dependency tracking failed for a %r event; "
                "this and later events may go unrecorded for this run",
                event,
                exc_info=True,
            )


def eval_environment(environment, **kwargs):
    if not environment:
        return ""

    def md5(string):
        return hashlib.md5(string.encode()).hexdigest()

    eval_globals = {"os": os, "sys": sys, "hashlib": hashlib, "md5": md5}
    eval_globals.update(kwargs)

    try:
        return str(eval(environment, eval_globals))
    except Exception as error:  # pylint: disable=broad-except
        return repr(error)


def process_result(result):
    failed = any(r.outcome == "failed" for r in result.values())
    duration = sum(value.duration for value in result.values())
    return {"failed": failed, "duration": duration}
