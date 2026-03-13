# Runs the decompiler to collect variable names from binaries containing
# debugging information, then strips the binaries and injects the collected
# names into that decompilation output.
# This generates an aligned, parallel corpus for training translation models.

# Requires Python 3

import argparse
import errno
import hashlib
import os
import subprocess
import tempfile as tf
import sys

from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from typing import Iterable, Tuple

# from runner import Runner

# COLLECT = os.path.join(dire_dir, "decompiler", "debug.py")
# DUMP_TREES = os.path.join(dire_dir, "decompiler", "dump_trees.py")


class Runner:
    dire_dir = os.path.dirname(os.path.abspath(__file__))
    COLLECT = os.path.join(dire_dir, "decompiler", "debug.py")
    DUMP_TREES = os.path.join(dire_dir, "decompiler", "dump_trees.py")

    def __init__(self, args: argparse.Namespace):
        self.ida = args.ida
        # prefer 64-bit IDA if present (older versions)
        base = os.path.basename(self.ida)
        dirn = os.path.dirname(self.ida)
        if base == "idat" and os.path.isfile(os.path.join(dirn, "idat64")):
            self.ida = os.path.join(dirn, "idat64")
        if base == "ida" and os.path.isfile(os.path.join(dirn, "ida64")):
            self.ida = os.path.join(dirn, "ida64")

        self.binaries_dir = args.binaries_dir
        self.output_dir = os.path.abspath(args.output_dir)
        self._num_files = args.num_files
        self.verbose = args.verbose
        self.num_threads = args.num_threads
        self.collect_timeout = args.collect_timeout
        self.dump_timeout = args.dump_timeout

        self.env = os.environ.copy()
        self.env["IDALOG"] = "/dev/stdout"
        self.env["OUTPUT_DIR"] = self.output_dir

        self.make_dir(self.output_dir)
        self.make_dir(os.path.join(self.output_dir, "types"))
        self.make_dir(os.path.join(self.output_dir, "bins"))
        self.logs_dir = os.path.join(self.output_dir, "ida_logs")
        self.make_dir(self.logs_dir)

        # Use RAM-backed memory for tmp if available
        if os.path.exists("/dev/shm"):
            tf.tempdir = "/dev/shm"
        try:
            self.run()
        except KeyboardInterrupt:
            pass

    @property
    def binaries(self) -> Iterable[Tuple[str, str]]:
        """Readable 64-bit ELFs in the binaries_dir and their paths"""

        def is_elf(root: str, path: str) -> bool:
            file_path = os.path.join(root, path)
            try:
                with open(file_path, "rb") as f:
                    header = f.read(5)
                    # '\x7fELF' means it's an ELF file, '\x02' means 64-bit
                    return header.startswith(b"\x7fELF")
            except IOError:
                return False

        return (
            (root, f)
            for root, _, files in os.walk(self.binaries_dir)
            for f in files
            if is_elf(root, f)
        )

    @property
    def num_files(self) -> int:
        """The number of files in the binaries directory. Note that this is not
        the total number of binaries because it does not check file headers. The
        number of binary files could be lower."""
        if self._num_files is None:
            self._num_files = 0
            for _, _, files in os.walk(self.binaries_dir):
                self._num_files += len(files)
        return self._num_files

    @staticmethod
    def make_dir(dir_path):
        """Make a directory, with clean error messages."""

        try:
            os.makedirs(dir_path)
        except OSError as e:
            if not os.path.isdir(dir_path):
                raise NotADirectoryError(f"'{dir_path}' is not a directory")
            if e.errno != errno.EEXIST:
                raise

    def run_decompiler(self, env, file_name, script, timeout=None, cwd=None, log_name=None, extra_args=None):
        """Run a decompiler script.

        Keyword arguments:
        file_name -- the binary to be decompiled
        env -- an os.environ mapping, useful for passing arguments
        script -- the script file to run (or 'python:...')
        timeout -- timeout in seconds (default no timeout)
        cwd -- working directory for the IDA process (default None)
        log_name -- persisted log file name to store under output_dir/ida_logs
        extra_args -- list of extra ida args
        """
        # capture IDA output to a log file we can print
        if log_name is None:
            log_name = "ida_run.log"
        log_path = os.path.join(self.logs_dir, log_name)
        # -A: auto-analysis, -B: batch mode, -L: write messages to file
        idacall = [self.ida, "-A", "-B", f"-L{log_path}"]
        if extra_args:
            idacall.extend(extra_args)
        # ensure absolute target path
        abs_target = os.path.abspath(file_name)
        idacall += [f"-S{script}", abs_target]
        if self.verbose:
            print(f"[IDA] cmd: {' '.join(idacall)}")
            print(f"[IDA] script: {script}")
            print(f"[IDA] file: {abs_target}")
            if cwd:
                print(f"[IDA] cwd: {cwd}")
        output = b""
        # set IDALOG for this call so logging surely goes to our file
        call_env = env.copy()
        call_env["IDALOG"] = log_path
        # IDA 9.0 on macOS requires HOME to be set
        if "HOME" not in call_env:
            call_env["HOME"] = os.path.expanduser("~")
        try:
            output = subprocess.check_output(
                idacall, env=call_env, timeout=timeout, stderr=subprocess.STDOUT, cwd=cwd
            )
            if self.verbose and output:
                print(output.decode("unicode_escape", errors="ignore"))
        except subprocess.TimeoutExpired as e:
            if self.verbose:
                print(f"[IDA] timeout after {timeout}s")
                if e.output:
                    print(e.output.decode("unicode_escape", errors="ignore"))
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"[IDA] returncode: {e.returncode}")
                if e.output:
                    print(e.output.decode("unicode_escape", errors="ignore"))
        finally:
            # always try to print ida log if available
            if self.verbose:
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                        log_txt = fh.read()
                    print("[IDA LOG BEGIN]\n" + log_txt + "\n[IDA LOG END]")
                except Exception:
                    pass
            # try to remove leftover db files
            subprocess.call(["rm", "-f", f"{abs_target}.i64"])

    def run_one(self, args: Tuple[str, str]) -> None:
        path, binary = args
        new_env = self.env.copy()
        print("starting")

        with tf.TemporaryDirectory() as tempdir:
            with tf.NamedTemporaryFile(dir=tempdir) as functions, tf.NamedTemporaryFile(
                dir=tempdir, delete=False
            ) as orig, tf.NamedTemporaryFile(dir=tempdir, delete=False) as stripped:

                abs_file_path = os.path.abspath(os.path.join(path, binary))

                # Copy binary to temp files for processing
                subprocess.check_output(["cp", abs_file_path, orig.name])
                subprocess.check_output(["cp", abs_file_path, stripped.name])

                # Strip debug info from the 'stripped' copy
                strip_cmd = []
                if sys.platform == "darwin":  # macOS
                    strip_cmd.append("/opt/homebrew/opt/llvm/bin/llvm-strip")
                    strip_cmd.append("--strip-debug")
                else:  # assume Linux/GNU
                    strip_cmd.append("strip")
                    strip_cmd.append("--strip-debug")
                strip_cmd.append(stripped.name)

                try:
                    subprocess.check_output(strip_cmd, stderr=subprocess.STDOUT)
                except FileNotFoundError as e:
                    # 'strip' command not found
                    if self.verbose:
                        print(f"[strip] command not found: {e.strerror}")
                    # Silently continue, maybe running on a system without dev tools
                    pass
                except subprocess.CalledProcessError as e:
                    # 'strip' command failed, probably because the binary is already stripped
                    if self.verbose:
                        output = e.output.decode(errors='ignore')
                        print(f"[strip] could not strip binary '{binary}' (already stripped?): {output.strip()}")
                    # Continue anyway, as the binary might already be stripped.
                    pass

                new_env["FUNCTIONS"] = functions.name
                # Build up hash string in 4k blocks
                file_hash = hashlib.sha256()
                with open(abs_file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        file_hash.update(byte_block)
                prefix = f"{file_hash.hexdigest()}_{binary}"
                new_env["PREFIX"] = prefix

                if os.path.exists(
                    os.path.join(self.output_dir, "bins", prefix + ".jsonl.gz")
                ):
                    if self.verbose:
                        print(f"{prefix} already collected, skipping")
                    return
                if os.path.exists(
                    os.path.join(self.output_dir, "types", prefix + ".jsonl.gz")
                ):
                    if self.verbose:
                        print(f"{prefix} types already collected, skipping")
                else:
                    if self.verbose:
                        print(f"[collect] starting for {prefix}")
                    # run collect on original binary (with debug info)
                    db_path = os.path.join(tempdir, f"{prefix}_collect.i64")
                    self.run_decompiler(
                        new_env,
                        orig.name,
                        self.COLLECT,
                        timeout=self.collect_timeout,
                        log_name=f"{prefix}_collect.log",
                        extra_args=["-c", f"-o{db_path}"],
                    )
                    if self.verbose:
                        print(f"[collect] finished for {prefix}")
                # Dump trees on stripped binary
                if self.verbose:
                    print(f"[dump] starting for {prefix}")
                db_path = os.path.join(tempdir, f"{prefix}_dump.i64")
                self.run_decompiler(
                    new_env,
                    stripped.name,
                    self.DUMP_TREES,
                    timeout=self.dump_timeout,
                    log_name=f"{prefix}_dump.log",
                    extra_args=["-c", f"-o{db_path}"],
                )
                if self.verbose:
                    print(f"[dump] finished for {prefix}")

    def run(self):
        # File counts for progress output

        # Create a temporary directory, since the decompiler makes a lot of
        # additional files that we can't clean up from here
        print(self.num_files)
        for binary in self.binaries:
            print(binary)
        with Pool(self.num_threads) as pool:
            for p in tqdm(
                    pool.imap_unordered(self.run_one, self.binaries),
                    total=self.num_files,
                    leave=True,
                    dynamic_ncols=True,
                    unit="bin",
                    smoothing=0.1,
            ):
                pass


parser = argparse.ArgumentParser(description="Run the decompiler to generate a corpus.")
parser.add_argument(
    "--ida",
    metavar="IDA",
    help="location of the idat64 binary",
    default="/home/jlacomis/bin/ida/idat64",
)
parser.add_argument(
    "-t",
    "--num-threads",
    metavar="N",
    help="number of threads to use",
    default=4,
    type=int,
)
parser.add_argument(
    "-n",
    "--num-files",
    metavar="N",
    help="number of binary files",
    default=None,
    type=int,
)
parser.add_argument(
    "-b",
    "--binaries_dir",
    metavar="BINARIES_DIR",
    help="directory containing binaries",
    required=True,
)
parser.add_argument(
    "-o", "--output_dir", metavar="OUTPUT_DIR", help="output directory", required=True,
)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument(
    "--collect-timeout",
    metavar="SECONDS",
    help="timeout for debug collection stage",
    default=120,
    type=int,
)
parser.add_argument(
    "--dump-timeout",
    metavar="SECONDS",
    help="timeout for dump trees stage",
    default=120,
    type=int,
)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    args = parser.parse_args()
    Runner(args)
