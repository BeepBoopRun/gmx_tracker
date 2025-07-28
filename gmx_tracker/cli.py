import argparse
import sys
from pathlib import Path
from .configs_parsing import parse_raw
from .simulation_handler import Simulation
from .track import run_simulations, SimulationGroupResult, SimulationStatus, SimulationResult
import os
import re
import logging
import time

BIG_MESSAGE_WIDTH = 40

def handle_commandline():
    parser = argparse.ArgumentParser(
        prog="GMX Tracker",
        description="Tool for real-time GROMACS performance measurement and tuning.",
    )

    parser.add_argument(
        "-p",
        "--try-paralell",
        action="store_true",
        help="Check to see if running multiple simulations in paralell improves performance.",
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
        "--run-best",
        type=int,
        metavar="N",
        help="TODO! Not implemented. Run most performant simulation N times to completion.",
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
        help="Give a file with newline delimited list of configurations to try. Additional syntax can be found in documentation.",
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
    # In practice, it'll probably mean find best solution, then try running it with nstlist=100 or nstlist=200 etc.
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
        help="Tries all combinations of cpu/gpu workload distribution.",
    )

    parser.add_argument(
        "--only-configurations",
        action="store_true",
        help="TODO! Add this back. Stops the program after printing configurations.",
    )

    args = parser.parse_args()

    return args




def get_numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else -1


def strip_trailing_numbers(s: str) -> str:
    return re.sub(r"\d+$", "", s)

def main():
    args = handle_commandline()

    if args.gromacs_command is None and args.use_file is None:
        print(
            "No input specified, please provide a file (-f) or a command to run (-c)!"
        )
        sys.exit(1)

    # TODO

    raw_input = ""
    if args.use_file is not None:
        with open(args.use_file, "r") as f:
            raw_input += f.read()

    if args.gromacs_command is not None:
        raw_input += "\n" # make sure that both are on different lines
        raw_input += args.gromacs_command
    
    simulation_pool = parse_raw(raw_input)

    print("CONFIGURATIONS TO TEST".center(BIG_MESSAGE_WIDTH, "-"))
    for id, configs in enumerate(simulation_pool):
        for sub_id, config in enumerate(configs):
            print(f"CONFIG {id}.{sub_id}: {" ".join(config)}")

    if args.only_configurations:
        return os.EX_OK

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

    print(f"RUNS DIR: {run_dir_path}")
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
    
    results: list[SimulationGroupResult] = [] 

    for id, configs in enumerate(simulation_pool):
        print(f"SIMULATION NR {id}".center(30, "-"))
        config_dir = run_dir_path.joinpath(f"CONFIG_{id}")
        config_dir.mkdir(parents=True, exist_ok=True)
        simulations: list[Simulation] = []
        for sub_id, config in enumerate(configs):
            run_dir = config_dir.joinpath(f"{sub_id}")
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"CONFIG {id}.{sub_id}: {" ".join(config)}")
            print(f"CONFIG {id}.{sub_id} DIR: {run_dir}")
            simulations.append(Simulation(
                gmx_arguments=config,
                run_dir=run_dir,
                necesarry_files=necesarry_sim_files,

            ))
        config_results = run_simulations(simulations, early_stop=args.early_stop, config_id=id)
        print(f"RESULT NR {id}".center(30, "-"))
        print(config_results)
        results.append(config_results)

    print("OVERALL RESULTS".center(BIG_MESSAGE_WIDTH, "-"))
    results = [res for res in results if res.overall_status == SimulationStatus.Success]
    results.sort(key=lambda x: x.total_performance, reverse=True)
    for res in results:
        print(f"Config ID: {res.config_id} Total performance: {res.total_performance} Total steps done: {res.total_steps}")


    