import logging
import os
import math
from datetime import datetime
from pathlib import Path
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)


# not available in older versions of python, hence reimplemented here
def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def minimum_performance_window(run_dir: Path) -> int | None:
    """Minimum difference of steps on which it makes sense to measure performance."""
    (root, dirs, files) = next(os.walk(run_dir, topdown=True))
    gmx_log = None
    for file in files:
        if file.endswith(".log"):
            gmx_log = os.path.join(root, file)

    if gmx_log is None:
        logger.warning(f"Couldn't find the logfile: {gmx_log}")
        return None

    calc_frequencies = {}
    with open(gmx_log, "r") as f:
        input_parameters_section = False
        for line in f.readlines():
            if "Input Parameters" in line:
                input_parameters_section = True

            if input_parameters_section is True and line == "\n":
                # this marks the end of the section, we don't
                # need to read more
                input_parameters_section = False

            if "nst" in line and "nsteps" not in line and input_parameters_section:
                in_line = line.split()
                if len(in_line) < 3:
                    continue
                if not in_line[2].isnumeric():
                    continue
                if in_line[2] != "0":
                    calc_frequencies[in_line[0]] = int(in_line[2])

            if "Overriding nsteps with value passed on the command line" in line:
                in_line = line.split()
                for word in in_line:
                    if word.isnumeric():
                        calc_frequencies["nstlist"] = int(word)
                break
    # print(calc_frequencies)
    if "nstlist" not in calc_frequencies:
        return None

    return lcm(calc_frequencies["nstlist"], 100)


def graph_performance(run_dir: Path, path_to_logfile, add=False):
    import matplotlib.pyplot as plt
    import numpy as np

    times = []
    steps = []
    with open(os.path.join(run_dir, path_to_logfile), "r") as f:
        for line in f.readlines():
            time, step = line.split(",")
            time = datetime.fromisoformat(time)
            step = int(step)
            print(time, step)
            times.append(time.timestamp())
            steps.append(step)

    perfs = np.diff(steps) / np.diff(times)

    plt.scatter(steps[:-1], perfs, s=2)
    plt.ylim(0, np.median(perfs) * 4)
    if not add:
        plt.show()


def parse_mdlog(log_file_or_run_dir: Path) -> dict[str,str] | None:
    """Creates a nested dictionary with key-value pairs from mdlog."""
    logfile = None
    if log_file_or_run_dir.suffix == ".log":
        logfile = log_file_or_run_dir
    else:
        for file in log_file_or_run_dir.iterdir():
            if file.suffix == ".log":
                logfile = file
    if logfile is None:
        print(f"Couldn't locate the logfile from path: {log_file_or_run_dir}")
        return None

    result = {}
    last_line = ""
    last_leading_whitespace = 0
    prev_access = []
    consume_next = False

    is_prolog = True
    is_citation = False

    with open(logfile, "r") as f:
        for line in f.readlines():

            if line == "\n":
                continue

            if is_prolog is True and line.startswith("GROMACS:"):
                is_prolog = False

            if line.startswith("++++"):
                is_citation = True

            if is_citation and line.startswith("--------"):
                is_citation = False
                continue

            if is_citation or is_prolog:
                continue

            leading_whitespace = len(line) - len(line.lstrip())
            if last_leading_whitespace == leading_whitespace and ":" in line:
                elems = [word.strip() for word in line.split(":")]
                # no words in line after ':', so take value from next line
                if len(elems) == 1:
                    print(line, end="")
                    consume_next = True
                else:
                    result[elems[0]] = elems[1]
            elif "=" in line:
                elems = [word.strip() for word in line.split("=")]

            last_leading_whitespace = leading_whitespace
            last_line = line
    
    return result

def get_stripped_mdlog(log_file_or_run_dir: Path) -> str|None:
    """Creates a nested dictionary with key-value pairs from mdlog."""
    logfile = None
    if log_file_or_run_dir.suffix == ".log":
        logfile = log_file_or_run_dir
    else:
        for file in log_file_or_run_dir.iterdir():
            if file.suffix == ".log":
                logfile = file
    if logfile is None:
        print(f"Couldn't locate the logfile from path: {log_file_or_run_dir}")
        return None

    stripped_mdlog = ""
    is_prolog = True
    is_citation = False

    with open(logfile, "r") as f:
        for line in f.readlines():

            if line == "\n":
                continue

            if is_prolog is True and line.startswith("GROMACS:"):
                is_prolog = False

            if line.startswith("++++"):
                is_citation = True

            if is_citation and line.startswith("--------"):
                is_citation = False
                continue

            if is_citation or is_prolog:
                continue

            stripped_mdlog += line
    
    return stripped_mdlog

class HardwareInformation(NamedTuple):
    cores: int
    threads: int
    GPUs: int


def get_hardware_information(run_dir: Path) -> HardwareInformation | None:
    logfile = None
    for file in run_dir.iterdir():
        if file.suffix == ".log":
            logfile = file
            break

    if logfile is None:
        logger.warning(f"Couldn't find runs md logfile in {run_dir}")
        return None

    with open(logfile, "r") as f:
        file_s = f.read()

        cpu_data = re.findall(r"(\d*) cores, (\d*) processing units", file_s)[0]
        gpu_data = re.findall(r"(\d*) compatible GPU", file_s)
    
    return HardwareInformation(cores=int(cpu_data[0]), threads=int(cpu_data[1]), GPUs=int(gpu_data[0]) if gpu_data else 0)



def get_failing_patterns(run_dir: Path) -> list[list[str]]:
    outfile = run_dir.joinpath("outfile.txt")
    if not outfile.exists():
        return []
    # TODO: Expand for more cases
    with open(outfile, "r") as f:
        for line in f.readlines():
            if "Update task can not run on the GPU" in line:
                return [["-update", "gpu"]]
    return []