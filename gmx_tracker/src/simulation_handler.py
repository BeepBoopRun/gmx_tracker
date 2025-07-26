import subprocess
import os
import threading
import csv
from datetime import datetime
from pathlib import Path
import logging
import threading
from typing import TypeVar
from rundir_analysis import *
from bisect import bisect_left,bisect_right

SECONDS_IN_DAY = 24 * 60 * 60
PICOSECONDS_IN_NANOSECOND = 1000

logger = logging.getLogger(__name__)


def setup_directory(necesarry_files: list[Path], destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for item in necesarry_files:
        symlink_path = destination.absolute() / item.name

        if symlink_path.exists():
            logger.warning(
                "Preparing a new simulation in a directory that is not empty!"
            )

        symlink_path.symlink_to(item)


class Simulation:
    """All components needed to run and manage a GROMACS simulation.

    Run and handles a running process, as well as a separate thread used to watch and measure performance of an ongoing simulation.
    """

    gmx_arguments: list[str]
    """List of all arguments needed to run a simulation, for example: ['gmx', 'mdrun', '-s', 'file.tpr']"""
    env: dict[str, str] | None
    """Enviromental variables with which to run the simulation."""
    perf_log: Path
    """Path to the logfile in which steps and their timestamps are saved."""
    watch_thread: threading.Thread | None
    """Thread that watches the running simulation."""
    gmx_process: subprocess.Popen | None
    """GROMACS process."""
    run_dir: Path
    """Directory in which the simulation is run."""
    last_tunepme_step: int | None
    """Last step in which pme tuning was seen, used to see if simulation has stopped tuning itself."""
    steps: list[int]
    """List of all steps measured by watching the output."""
    times: list[datetime]
    """List of all the times when steps were measured."""
    verbose: bool
    """Decides if simulation will output to screen."""

    current_step: int | None
    sim_timestep: float | None
    has_printed: bool
    """Useful to see if GROMACS silently crashed."""

    def __init__(
        self,
        gmx_arguments: list[str],
        run_dir: Path,
        necesarry_files: list[Path] = [],
        env=None,
        perf_log: Path = Path("perf_log.csv"),
        verbose=False,
    ):
        self.gmx_arguments = gmx_arguments
        self.env = env
        self.perf_log = perf_log
        self.run_dir = run_dir
        self.verbose = verbose

        self.watch_thread = None
        self.gmx_process = None
        self.last_tunepme_step = None
        self.sim_timestep = None
        self.has_printed = False
        self.steps = []
        self.times = []

        if necesarry_files:
            setup_directory(necesarry_files=necesarry_files, destination=run_dir)

    def start(self):
        # moves temporarily to run directory, so gmx arguments refer to files within it
        original_cwd = os.getcwd()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.run_dir)

        process = subprocess.Popen(
            self.gmx_arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=self.env,
        )

        thread = threading.Thread(
            target=self.read_and_timestamp_output,
            args=(process.stdout,),
        )
        thread.start()
        self.watch_thread = thread
        self.gmx_process = process

        os.chdir(original_cwd)

        return self

    def read_and_timestamp_output(self, pipe, save_out=True):
        # buffering=1 means we can plot in real-time
        # as we buffer on per-line basis
        csvhandle = open(self.perf_log, "w", buffering=1)
        csv_writer = csv.writer(csvhandle)
        outfile_handle = open(os.path.join(self.run_dir, "outfile.txt"), "w",  buffering=1)

        for line in pipe:
            if save_out:
                outfile_handle.write(line)
            self.has_printed = True
            timestamp = datetime.now()
            words = line.split()

            if len(words) >= 2 and words[0] == "step":
                if self.verbose:
                    # who needs whitespace?
                    print(line.strip())
                step_value = words[1]
                while not step_value[-1].isnumeric():
                    step_value = step_value[:-1]

                step_value = int(step_value)
                csv_writer.writerow([timestamp, step_value])
                self.times.append(timestamp)
                self.steps.append(step_value)

                if "timed with pme" in line:
                    self.last_tunepme_step = step_value


        csvhandle.close()
        outfile_handle.close()

    def wait(self) -> int | None:
        if self.gmx_process is None or self.watch_thread is None:
            logger.info("Simulation wasn't run! Cannot wait on what hasn't started...")
            return
        returncode = self.gmx_process.wait()
        self.watch_thread.join()
        return returncode

    def kill(self):
        if self.gmx_process is None or self.watch_thread is None:
            logger.info(
                "Simulation wasn't run! Cannot kill a simulation that isn't running..."
            )
            return
        self.gmx_process.kill()
        self.watch_thread.join()

    def graph_performance(self, add=False):
        graph_performance(self.run_dir, self.perf_log, add)

    def is_tuned(self) -> bool:
        # check if simulation has properly started
        # TODO: Account for dynamic-load balancing

        if (
            self.last_tunepme_step is None
            and len(self.steps) > 2
            and self.steps[-1] > 1000
            and (datetime.now() - self.times[0]).total_seconds() > 20
        ):
            return True

        if self.last_tunepme_step is None or self.steps is None:
            return False

        if "-notunepme" in self.gmx_arguments:
            return True

        return self.last_tunepme_step < self.steps[-1]

    def timestep(self) -> float | None:

        if self.sim_timestep is not None:
            return self.sim_timestep

        gmx_log = None
        for file in self.run_dir.iterdir():
            if file.suffix == ".log":
                gmx_log = file
                break

        if gmx_log is None:
            logger.warning(f"Couldn't find the logfile: {gmx_log}")
            return None

        dt = None
        with open(gmx_log, "r") as f:
            input_parameters_section = False
            for line in f.readlines():
                if "Input Parameters" in line:
                    input_parameters_section = True

                if input_parameters_section is True and line == "\n":
                    return dt

                if input_parameters_section and "dt" in line:
                    return float(line.split()[2])

    def current_performance(self, last_nsteps: int) -> float | None:
        """Calculate performance based on last n steps of the simulation."""
        if self.steps is None or self.times is None:
            return None
        if self.steps[-1] - last_nsteps < 0:
            last_nsteps = self.steps[0]
        start_step = self.steps[-1] - last_nsteps
        # if doesn't exist, find lowest value greater than it
        if self.find_step(start_step) is None:
            i = bisect_right(self.steps, start_step)
            if i == len(self.steps):
                return None
            else:
                start_step = self.steps[i]
        return self.performance(start_step=start_step, end_step=self.steps[-1])

    def performance(self, start_step: int, end_step: int) -> float | None:
        """Calculate performance between given steps, returns None if they don't exist."""


        idx_start_step = self.find_step(start_step)
        idx_end_step = self.find_step(end_step)

        timestep = self.timestep()
        if timestep is None:
            return None

        if idx_start_step is None or idx_end_step is None:
            logger.warning(f"Couldn't find steps needed to computer performance, steps used: ({start_step},{end_step})")
            return None

        start_time = self.times[idx_start_step]
        end_time = self.times[idx_end_step]

        if end_time <= start_time or end_step <= start_step:
            return None

        return (
            ((end_step - start_step) / (end_time - start_time).total_seconds())
            * timestep
            / PICOSECONDS_IN_NANOSECOND
            * SECONDS_IN_DAY
        )

    def find_step(self, step: int) -> int | None:
        """Returns index of the step in step list or None if not found."""
        if (
            self.steps is None
            or self.times is None
            or self.steps[-1] < step
            or step < 0
        ):
            return None
        
        i = bisect_left(self.steps, step)
        if i != len(self.steps) and self.steps[i] == step:
            return i
        else:
            return None
