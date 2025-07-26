# TODO: Account for load balancing hard. Suprisingly tricky, as it doesn't always show up.

from datetime import datetime
import os
import re
import time
from pathlib import Path
import argparse
import logging
import sys
import csv
import itertools
from enum import Enum

# import matplotlib.pyplot as plt
# import numpy as np
from datetime import datetime
from typing import NamedTuple
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from simulation_handler import Simulation
from rundir_analysis import minimum_performance_window, get_hardware_information, get_failing_patterns

NUMBER_OF_BEST_TO_TRY = 2
BIG_MESSAGE_WIDTH = 40
SECONDS_IN_DAY = 24 * 60 * 60
PICOSECONDS_IN_NANOSECOND = 1000
REFRESH_TIME = 5
ROUND_PRECISION = 2
WAITING_TIME = 60
MAXIMUM_PERCENT_CHANGE = 0.03
# based on GROMACS 2025.1
ILLEGAL_PATTERNS = [
    ["-pme", "cpu", "-pmefft", "gpu"],
    ["-nb", "cpu", "-bonded", "gpu"],
    ["-pme", "gpu", "-nb", "cpu"]
]


logger = logging.getLogger(__name__)


class SimulationStatus(Enum):
    # TODO: More could be added depending on reason of failure
    Success = 0
    Failure = 1


class SimulationResult(NamedTuple):
    """Represents a person with basic contact info.

    Attributes:
        run_id (str)
        run_sub_id (str): Used when there was more than one simulation run in paralell.
        status (SimulationStatus): Informs if the run was a success.
        gmx_arguments (str): Command line call to start the simulation.
        performance (float): Performance of the simulation in [ns/day]
        run_dir (Path): Absolute Path to directory where simulation ran.
    """

    run_id: int
    run_sub_id: int
    status: SimulationStatus
    gmx_arguments: str
    performance: float
    steps_done: int
    run_dir: Path

    def __str__(self):
        return f"""\
Run ID: {self.run_id}
Run sub-ID: {self.run_sub_id} 
Status: {self.status.name}
GROMACS arguments: {self.gmx_arguments}
Measured performance: {self.performance} [ns/day] 
Steps done: {self.steps_done}"""


# Probably merging info from individual runs would make this better.
class SimulationGroupResult(NamedTuple):
    config_id: int
    status: SimulationStatus
    total_performance: float
    total_steps_done: int
    run_dir: Path

    def __str__(self):
        return f"""\
Run ID: {self.config_id}
Status: {self.status.name}
Total measured performance: {self.total_performance} [ns/day] 
Total steps done: {self.total_steps_done}"""


def handle_commandline():
    parser = argparse.ArgumentParser(
        prog="GMX Tracker",
        description="This program is used to find the optimal performance (ns/day) configuration for your GROMACS simulation.",
        epilog="Bottom 3",
    )

    parser.add_argument(
        "-p",
        "--allow-paralell",
        action="store_true",
        help="Check to see if running multiple simulations improves performance, can help with time to nanosecond.",
    )
    parser.add_argument(
        "-c",
        "--gromacs-command",
        help="""Initial gmx mdrun command, for example: 'gmx mdrun -s file.tpr'.
                        Syntax like 'gmx mdrun -s file.tpr -pme cpu,gpu' is additionaly supported.
                        It means that both values, here 'cpu' and 'gpu' will be tried for the argument '-pme'""",
    )
    parser.add_argument(
        "-n",
        "--run-multiple",
        type=int,
        metavar="N",
        help="TODO! Not implemented. Number of simulations run to completion, can be used in conjuction with -p to quickly get replicas.",
    )
    parser.add_argument(
        "-e",
        "--early-stop",
        action="store_true",
        help="With this option, simulation stops when it's performance reaches a steady state, instead of running till the end.",
    )
    parser.add_argument(
        "-f",
        "--use-file",
        type=Path,
        metavar="FILE",
        help="Give a file with newline delimited list of configurations to try.",
    )
    parser.add_argument(
        "-d",
        "--sim-directory",
        type=Path,
        help="Specify path to directory with simulation files, by default working directory is assumed.",
    )
    parser.add_argument(
        "-r",
        "--runs-directory",
        type=Path,
        help="Specify path to directory where runs should be saved, by default working directory is assumed.",
    )
    # TODO: Implement
    parser.add_argument(
        "-l",
        "--nstlist-tuning",
        action="store_true",
        help="TODO! Not implemented. Allows for tuning of the nstlist, this could have consequences in specific conditions!",
    )
    parser.add_argument(
        "-x",
        "--exhaustive-tuning",
        action="store_true",
        help="All flags that help with tuning will be tried.",
    )

    return parser.parse_args()


def get_numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else -1


def strip_trailing_numbers(s: str) -> str:
    return re.sub(r"\d+$", "", s)


# using a single list since few values and who needs extra imports...
get_available_tuning_options_cache = []


def get_available_tuning_options(args_to_call_mdrun: list[str]) -> list[str] | None:

    for cache_pair in get_available_tuning_options_cache:
        if cache_pair[0] == args_to_call_mdrun:
            return cache_pair[1]

    # try two times to get the list of available arguments
    result = None
    for _ in range(2):
        result = subprocess.run(
            args_to_call_mdrun + ["-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode == 0:
            break

    if result is None or result.returncode != 0:
        logging.warning(f"Could not get available flags from {args_to_call_mdrun} -h!")
        if result is None:
            logging.warning("Process wasn't run! Maybe an exception?")
        else:
            logging.warning(result.stderr)
        return None

    last_line = ""
    args = []
    for line in result.stdout.splitlines():
        if "cpu, gpu" in line:
            args.append(last_line.split()[0])
        last_line = line

    get_available_tuning_options_cache.append([args_to_call_mdrun, args])
    print(f"For version: {stringify_arguments(args_to_call_mdrun)}")
    print(f"Available arguments: {stringify_arguments(args)}")

    return args


def get_all_simulation_configs(
    arguments_list: list[str | list[str]],
) -> list[list[str]]:
    """Uses a list of necesarry arguments, with nested lists where multiple options are possible.

    For example: ['gmx', 'mdrun', '-s', 'file.tpr', '-pme', ['cpu', 'gpu']] -> [[..., '-pme', 'cpu'], [..., '-pme', 'gpu']]
    """
    simulation_pool = [[]]
    for arg in arguments_list:
        if type(arg) is not list:
            for s in simulation_pool:
                s.append(arg)
        else:
            option_count = len(arg)
            simulation_pool = [
                s.copy() for s in simulation_pool for _ in range(option_count)
            ]
            for i, s in enumerate(simulation_pool):
                s.append(arg[i % option_count])
    return simulation_pool


def sort_simulations(simulation_list: list[list[str]]) -> list[list[str]]:
    """Sorts the simulations most to least promising

    NOTE: The implementation is crude. Much to improve here."""
    # number of threads, what exactly is on the gpu etc. could make this much better.
    # more GPU, more speed
    return sorted(simulation_list, key=lambda sim: sim.count("gpu"), reverse=True)


def contains_pattern(sim: list[str], pattern: list[str]):
    i = 0
    contains_pattern = True
    while i < len(pattern):
        contains_pattern = False
        contains_argument = (
            pattern[i].startswith("-")
            and i < len(pattern) - 1
            and not pattern[i + 1].startswith("-")
        )
        if contains_argument:
            for j in range(len(sim) - 1):
                if sim[j : j + 2] == pattern[i : i + 2]:
                    contains_pattern = True
                    break
        else:
            for j in range(len(sim)):
                if sim[j] == pattern[i]:
                    contains_pattern = True
                    break
        if contains_pattern is False:
            break
        if contains_argument:
            i += 2
        else:
            i += 1
    return contains_pattern


def filter_simulations(
    simulation_list: list[list[str]], patterns_to_remove: list[list[str]]
) -> list[list[str]]:
    result = []

    for sim in simulation_list:
        fully_matches_any_pattern = any(
            [contains_pattern(sim, p) for p in patterns_to_remove]
        )
        if not fully_matches_any_pattern:
            result.append(sim)
    return result


def deduplicate_simulations(simulation_list: list[list[str]]) -> list[list[str]]:
    result = []
    for sim in simulation_list:
        # if both are true, they must be identical
        # inefficient, but this is a very small part of the program
        if not any(
            [
                contains_pattern(sim, pattern) and contains_pattern(pattern, sim)
                for pattern in result
            ]
        ):
            result.append(sim)
    return result


def parse_simulation(raw_input: str | list[str]) -> list[list[str]]:
    args = raw_input if isinstance(raw_input, list) else raw_input.split()

    result = []
    for arg in args:
        if "|" in arg:
            arg = arg.split("|")
        result.append(arg)
    return get_all_simulation_configs(result)


def remove_argument(simulation: list[str], argument: str) -> list[str]:
    # assumes -arg value
    if contains_pattern(simulation, [argument]):
        for i in range(len(simulation)):
            if simulation[i] == argument:
                simulation = simulation[:i] + simulation[i + 2 :]
    return simulation


# suprisingly nontrivial haha, needs further testing
def specify_hardware_usage(
    simulation: list[str],
    OpenMPthreads: int | None = None,
    pin_offset: int | None = None,
    gpu_ids: list[int] | None = None,
):

    gpu_count = len(gpu_ids) if gpu_ids else None

    if OpenMPthreads is not None:
        assert OpenMPthreads > 0
        simulation = remove_argument(simulation, "-ntomp")
        simulation += ["-ntomp", str(OpenMPthreads//(gpu_count or 1))]

    if pin_offset is not None:
        assert pin_offset >= 0
        simulation = remove_argument(simulation, "-pin")
        simulation += ["-pin", "on"]
        simulation = remove_argument(simulation, "-pinoffset")
        simulation += ["-pinoffset", str(pin_offset)]

    if gpu_ids is not None:
        simulation = remove_argument(simulation, "-gpu_id")
        simulation = remove_argument(simulation, "-ntmpi")
        if gpu_count:
            simulation += ["-ntmpi", str(gpu_count)]
            if contains_pattern(sim=simulation, pattern=["-pme", "gpu"]):
                if gpu_count > 1:
                    simulation += ["-npme", str((gpu_count // 4) or 1)]

        simulation += ["-gpu_id", ",".join([str(gpu) for gpu in gpu_ids])]

    env = os.environ
    env["CUDA_VISIBLE_DEVICES"]= (",".join([str(gpu_id) for gpu_id in gpu_ids]) if gpu_ids is not None else "")
    return (simulation,env)

def stringify_arguments(args: list[str]) -> str:
    return " ".join(args)

def running_sim_summary(sim: Simulation, run_id: int, run_sub_id: int = 0) -> None:
    tune_status = "Finished" if sim.is_tuned() else "In progress"
    format = lambda x: round(x, ROUND_PRECISION) if x is not None else "NA"
    optimal_window = minimum_performance_window(sim.run_dir)
    relative_perf = None
    if (
        optimal_window is not None
        and len(sim.steps) > 0
        and sim.steps[-1] >= 2 * optimal_window
    ):
        prev_perf1 = sim.performance(sim.steps[-1] - optimal_window, sim.steps[-1])
        prev_perf2 = sim.performance(
            sim.steps[-1] - 2 * optimal_window,
            sim.steps[-1] - optimal_window,
        )
        if prev_perf2 is not None and prev_perf1 is not None:
            relative_perf = f"{round(100 * (prev_perf1/prev_perf2 - 1), 3)}%"

    print(
        f"{run_id}.{run_sub_id}: Steps done: {sim.steps[-1]}, Tuning status: {tune_status}, Performance last (200, 500, 1000) steps: "
        f"({format(sim.current_performance(200))}, {format(sim.current_performance(500))}, {format(sim.current_performance(1000))}) [ns/day], "
        f"Relative change in perf: {relative_perf if relative_perf is not None else 'NA'}"
    )


def stop_policy(sim: Simulation):
    """Defines the early stop condition; point when we should stop the simulation, since we run enough to estimate performance."""
    # TODO: Could be changed, improved and all else.

    window_size = minimum_performance_window(sim.run_dir)
    if sim.steps is None or window_size is None or sim.is_tuned() is False:
        return False

    if (
        sim.last_tunepme_step is not None
        and sim.steps[-1] < sim.last_tunepme_step + 2 * window_size
    ):
        return False

    perf1 = sim.performance(
        start_step=sim.steps[-1] - window_size, end_step=sim.steps[-1]
    )
    perf2 = sim.performance(
        start_step=sim.steps[-1] - 2 * window_size,
        end_step=sim.steps[-1] - window_size,
    )

    if perf1 is None or perf2 is None:
        return False

    ratio = perf1 / perf2

    return ratio > 1 - MAXIMUM_PERCENT_CHANGE and ratio < 1 + MAXIMUM_PERCENT_CHANGE

def main():
    args = handle_commandline()

    if args.gromacs_command is None and args.use_file is None:
        logger.error(
            "No input specified, please provide a file (-f) or a command to run (-c)!"
        )
        sys.exit(1)

    simulation_pool = []

    if args.gromacs_command is not None:
        simulation_pool.extend(parse_simulation(args.gromacs_command))
    if args.use_file is not None:
        with open(args.use_file, "r") as f:
            for line in f.readlines():
                if line.startswith("#") or line.isspace():
                    continue
                simulation_pool.extend(parse_simulation(line))

    remove_patterns = [pattern[1:] for pattern in simulation_pool if pattern[0] == "!"]
    simulation_pool = [sim for sim in simulation_pool if sim[0] != "!"]

    if args.exhaustive_tuning:
        print("PREPARING CONFIGURATIONS".center(BIG_MESSAGE_WIDTH, "-"))
        for sim in simulation_pool:
            arguments_to_call_mdrun = []
            for arg in sim:
                arguments_to_call_mdrun.append(arg)
                if arg == "mdrun":
                    break

            all_arguments = get_available_tuning_options(arguments_to_call_mdrun)
            if all_arguments is not None:
                for arg in all_arguments:
                    if arg not in sim:
                        sim.append(arg)
                        sim.append(["cpu", "gpu"])
            else:
                logger.warning("Could not find the tuning options!")
                print(
                    "Couldn't get all tuning options, using only those supplied by user..."
                )
        new_pool = []
        for sim in simulation_pool:
            new_pool.extend(get_all_simulation_configs(sim))
        simulation_pool = new_pool

    if remove_patterns:
        simulation_pool = filter_simulations(simulation_pool, remove_patterns)

    simulation_pool = filter_simulations(simulation_pool, ILLEGAL_PATTERNS)

    # deleting tags
    simulation_pool = [
        [arg for arg in sim if not arg.startswith("#")] for sim in simulation_pool
    ]

    simulation_pool = deduplicate_simulations(simulation_pool)

    # sort them most promising to least
    # could be used to improve early stop
    simulation_pool = sort_simulations(simulation_pool)

    print("CONFIGURATIONS TO TEST".center(BIG_MESSAGE_WIDTH, "-"))
    for i, s in enumerate(simulation_pool):
        print(f"CONFIG {i}: {stringify_arguments(s)}")

    if args.sim_directory:
        sim_files_dir = Path(args.sim_directory).absolute()
    else:
        sim_files_dir = Path(os.getcwd()).absolute()
    necesarry_sim_files = [x for x in sim_files_dir.iterdir()]

    print("SIMULATION DIRECTORY".center(BIG_MESSAGE_WIDTH, "-"))
    print(f"SIM DIR: {sim_files_dir}")

    print("RUNS DIRECTORY".center(BIG_MESSAGE_WIDTH, "-"))

    if args.runs_directory:
        run_dir_path = Path(args.runs_directory)
    else:
        # default name for runs library... runs!
        run_dir_path = Path(os.path.join(os.getcwd(), "runs"))

    if run_dir_path.is_dir():
        is_used = False

        for subdir in run_dir_path.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith("RUN_"):
                is_used = True
        if is_used:
            print(
                "Chosen runs directory already contains runs inside! "
                "Changing directory to avoid collision..."
            )

            stripped_basename = strip_trailing_numbers(run_dir_path.name)

            # handles files as well
            similar_dirs = [
                dir
                for dir in run_dir_path.parent.iterdir()
                if dir.name.startswith(stripped_basename)
            ]
            max_number_suffix = max(
                [get_numeric_suffix(dir.name) for dir in similar_dirs]
            )

            run_dir_path = run_dir_path.parent.joinpath(
                f"{stripped_basename}_{max_number_suffix+1}"
            )

    print(f"RUN DIR: {run_dir_path}")
    run_dir_path.mkdir(parents=True, exist_ok=True)

    cli_log = run_dir_path.joinpath("cli.log")
    cli_log.touch(exist_ok=True)
    logging.basicConfig(
        filename=cli_log,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simulation_results: list[SimulationResult] = []
    final_id = len(simulation_pool)

    with open(run_dir_path.joinpath("results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SimulationResult._fields)

    failing_patterns: list[list[str]] = []
    # MAIN LOOP
    for id, sim_config in enumerate(simulation_pool):
        current_run_dir = run_dir_path.joinpath(f"RUN_{id}")

        print(f"SIMULATION NR {id}".center(30, "-"))
        
        failed_by_deault = filter_simulations([sim_config], failing_patterns) == [] 

        if not failed_by_deault:
            print(f"RUN DIRECTORY: {current_run_dir}")

        print(f"CONFIGURATION: {stringify_arguments(sim_config)}")

        # we know it will fail if it contains any of the patterns that suggest failure
        if failed_by_deault:
            print("Failed based on previous runs.")
            continue

        sim = Simulation(
            gmx_arguments=sim_config,
            run_dir=current_run_dir,
            necesarry_files=necesarry_sim_files,
        )

        start_time = datetime.now()

        sim.start()
        last_step = None
        assert sim.gmx_process is not None
        while sim.gmx_process.poll() is None and not failed_by_deault:

            time.sleep(REFRESH_TIME)

            if sim.has_printed == False:
                print("Simulation not yet started.")
                if (datetime.now() - start_time).total_seconds() > WAITING_TIME:
                    sim.kill()
            elif len(sim.steps) > 0 and last_step != sim.steps[-1]:
                running_sim_summary(sim, run_id=id)
                last_step = sim.steps[-1]

            if args.early_stop and stop_policy(sim):
                sim.kill()
            

        status = (
            SimulationStatus.Success if len(sim.steps) > 0 else SimulationStatus.Failure
        )

        if status == SimulationStatus.Failure:
            # TODO: Doesn't consider case of multiple executables.
            pattern = get_failing_patterns(sim.run_dir)
            failing_patterns.extend(pattern)
            logger.info(f"Simulation fail, responsible pattern: {pattern}")

        optimal_window = minimum_performance_window(sim.run_dir)
        # for simulations that didn't run, we report performance of 0
        final_perf = 0.0
        if optimal_window is not None and len(sim.steps) > 0:
            final_perf = (
                sim.performance(
                    sim.steps[-1] - optimal_window * 2 - 100, sim.steps[-1] - 100
                )
                or 0.0
            )

        simulation_results.append(
            SimulationResult(
                run_id=id,
                run_sub_id=0,
                status=status,
                gmx_arguments=stringify_arguments(sim_config),
                performance=round(final_perf, ROUND_PRECISION),
                steps_done=sim.steps[-1] if len(sim.steps) > 0 else 0,
                run_dir=sim.run_dir,
            )
        )
        print(f"RUN RESULTS".center(30, "-"))
        print(f"{simulation_results[-1]}\n")
        if simulation_results[-1].status == SimulationStatus.Success:
            with open(run_dir_path.joinpath("results.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(simulation_results[-1])

    successful_simulations = [
        sim for sim in simulation_results if sim.status == SimulationStatus.Success
    ]

    print("RESULTS".center(BIG_MESSAGE_WIDTH, "-"))
    successful_simulations.sort(key=lambda x: x.performance, reverse=True)
    for sim in successful_simulations:
        print(f"{sim}\n")

    if not args.allow_paralell or not successful_simulations:
        sys.exit(0)

    hard_info = get_hardware_information(successful_simulations[0].run_dir)
    if hard_info is None:
        logging.warning(
            f"Couldn't find hardware info in directory: {successful_simulations[0].run_dir}!"
        )
        sys.exit(1)
    # print(f"Got info: {hard_info}")

    # MAIN LOOP PARALELL
    sims_paralell = hard_info.GPUs if hard_info.GPUs > 1 else 2
    better_performance = True
    simulation_results_paralell: list[list[SimulationResult]] = []
    simulation_group_results: list[SimulationGroupResult] = []
    last_runs_id = len(simulation_pool)
    while better_performance:
        with open(
            run_dir_path.joinpath(f"results{sims_paralell}.csv"), "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(SimulationResult._fields)
        with open(
            run_dir_path.joinpath(f"results{sims_paralell}_grouped.csv"),
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(SimulationGroupResult._fields)



        print(f"{sims_paralell} RUNS PARALELL".center(BIG_MESSAGE_WIDTH, "-"))
        configurations = [
            config
            for config in itertools.combinations_with_replacement(
                successful_simulations[:NUMBER_OF_BEST_TO_TRY], r=sims_paralell
            )
        ]

        print("CONFIGURATIONS TO TEST".center(BIG_MESSAGE_WIDTH, "-"))
        for i, s in enumerate(configurations):
            for j, x in enumerate(s):
                print(f"CONFIG {i+last_runs_id}.{j}: {x.gmx_arguments}")

        for raw_id, config in enumerate(configurations):
            config_id = raw_id + last_runs_id
            sims: list[Simulation] = []
            parent_run_dir = run_dir_path.joinpath(f"RUN_{config_id}")
            print(f"SIMULATION NR {config_id}".center(30, "-"))

            pin_offset = 0
            gpu_offset = 0
            for run_id, sim_conf in enumerate(config):
                run_dir = parent_run_dir.joinpath(str(run_id))
                gmx_args = parse_simulation(sim_conf.gmx_arguments)[0]
                available_threads = hard_info.threads // sims_paralell
                # spread the remainder to use all resources
                if hard_info.threads % sims_paralell > run_id:
                    available_threads += 1
                available_gpus = hard_info.GPUs // sims_paralell
                if hard_info.GPUs % sims_paralell > run_id:
                    available_gpus += 1
                gpu_ids = (
                    list(range(gpu_offset, gpu_offset + available_gpus))
                    if available_gpus > 0
                    else None
                )
                gmx_args,gmx_env = specify_hardware_usage(
                    simulation=gmx_args,
                    OpenMPthreads=available_threads,
                    pin_offset=pin_offset,
                    gpu_ids=gpu_ids,
                )

                pin_offset += available_threads
                gpu_offset += available_gpus
                sim = Simulation(
                    gmx_arguments=gmx_args,
                    env=gmx_env,
                    run_dir=run_dir,
                    necesarry_files=necesarry_sim_files,
                )
                sims.append(sim)

                print(f"RUN DIRECTORY {run_id}: {sim.run_dir}")
                print(
                    f"CONFIGURATION {run_id}: {stringify_arguments(sim.gmx_arguments)}"
                )

            for sim in sims:
                sim.start()

            start_time = datetime.now()
            sims_info: list[dict[str, int]] = [{"last_step": 0}.copy() for sim in sims]

            assert all([sim.gmx_process is not None for sim in sims])
            while any([sim.gmx_process.poll() is None for sim in sims]):

                time.sleep(REFRESH_TIME)

                for run_sub_id, (sim, sim_info) in enumerate(zip(sims, sims_info)):
                    if sim.has_printed == False:
                        print(f"{config_id}.{run_sub_id}: Simulation not yet started.")
                        if (datetime.now() - start_time).total_seconds() > WAITING_TIME:
                            sim.kill()
                    elif len(sim.steps) > 0 and sim_info["last_step"] != sim.steps[-1]:
                        running_sim_summary(
                            sim, run_id=config_id, run_sub_id=run_sub_id
                        )
                        sim_info["last_step"] = sim.steps[-1]
                if (
                    args.early_stop
                    and all([stop_policy(sim) for sim in sims])
                    or any([sim.gmx_process.poll() is not None for sim in sims])
                ):
                    for sim in sims:
                        sim.kill()

            config_run_results: list[SimulationResult] = []
            for run_sub_id, sim in enumerate(sims):
                optimal_window = minimum_performance_window(sim.run_dir)
                final_perf = 0.0
                if optimal_window is not None and len(sim.steps) > 0:
                    final_perf = (
                        sim.performance(
                            sim.steps[-1] - optimal_window * 2 - 100,
                            sim.steps[-1] - 100,
                        )
                        or 0.0
                    )
                status = (
                    SimulationStatus.Success
                    if all([len(sim.steps) > 0 for sim in sims])
                    else SimulationStatus.Failure
                )

                sim_result = SimulationResult(
                    run_id=config_id,
                    run_sub_id=run_sub_id,
                    status=status,
                    gmx_arguments=stringify_arguments(sim.gmx_arguments),
                    performance=final_perf,
                    steps_done=(
                        sim.steps[-1] if status == SimulationStatus.Success else 0
                    ),
                    run_dir=sim.run_dir,
                )

                config_run_results.append(sim_result)
                if config_run_results[-1].status == SimulationStatus.Success:
                    with open(
                        run_dir_path.joinpath(f"results{sims_paralell}.csv"),
                        "a",
                        newline="",
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(config_run_results[-1])

            print(f"INDIVIDUAL RUN RESULTS".center(30, "-"))
            for sim in config_run_results:
                print(f"{sim}\n")

            print(f"GROUP RUN RESULTS".center(30, "-"))
            total_performance = sum([sim.performance for sim in config_run_results])
            total_steps = sum([sim.steps_done for sim in config_run_results])
            final_status = (
                SimulationStatus.Success
                if all(
                    [
                        sim.status is SimulationStatus.Success
                        for sim in config_run_results
                    ]
                )
                else SimulationStatus.Failure
            )

            group_result = SimulationGroupResult(
                config_id=config_id,
                status=final_status,
                total_performance=total_performance,
                total_steps_done=total_steps,
                run_dir=parent_run_dir,
            )
            print(group_result)

            if group_result.status == SimulationStatus.Success:
                with open(
                    run_dir_path.joinpath(f"results{sims_paralell}_grouped.csv"),
                    "a",
                    newline="",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(group_result)

            simulation_group_results.append(group_result)
            simulation_results_paralell.append(config_run_results)
            config_id += 1

        better_performance = max(
            [res.total_performance for res in simulation_group_results]
        ) > max([sim.performance for sim in simulation_results])
        simulation_group_results.sort(key=lambda x: x.total_performance, reverse=True)
        print(f"{sims_paralell} PARALELL RUNS RESULTS".center(30, "-"))
        for res in simulation_group_results:
            print(res)
            print("")

        sims_paralell += 1
    print("RESULTS OVERALL".center(BIG_MESSAGE_WIDTH, "-"))
    all_results = simulation_group_results + successful_simulations
    all_results.sort(
        key=lambda x: (
            x.performance if isinstance(x, SimulationResult) else x.total_performance
        ),
        reverse=True,
    )
    for res in all_results:
        if res.status == SimulationStatus.Success:
            print(res)
            print("")

if __name__ == "__main__":
    main()