import subprocess
import logging
import itertools


logger = logging.getLogger(__name__)

# using a single list since few values and who needs extra imports...
get_available_tuning_options_cache = []


def get_available_tuning_options(
    args_to_call_mdrun: list[str], print_info: bool = True
) -> list[str] | None:

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
    gmx_args = []
    for line in result.stdout.splitlines():
        if "cpu, gpu" in line:
            gmx_args.append(last_line.split()[0])
        last_line = line

    get_available_tuning_options_cache.append([args_to_call_mdrun, gmx_args])
    if print_info:
        print(f"For version: {" ".join(args_to_call_mdrun)}")
        print(f"Available arguments: {" ".join(gmx_args)}")

    return gmx_args


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


def parse_raw(raw_input: str) -> list[list[list[str]]]:

    tidy_input: list[str] = []
    # remove comments and line breaks
    buffer = ""
    for line in raw_input.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.endswith("\\"):
            line = line[:-1]
            buffer += line
            continue
        if buffer:
            buffer += line
            tidy_input.append(buffer)
            buffer = ""
            continue
        tidy_input.append(line)

    remove_patterns: list[str] = []

    # group paralell
    grouped: list[list[str]] = []
    group = []
    for i in range(len(tidy_input)):
        if tidy_input[i].startswith("@"):
            group.append(tidy_input[i][1:])
            continue
        if group:
            grouped.append(group)
            group = []

        if tidy_input[i] == "":
            continue
        if tidy_input[i].startswith("!"):
            remove_patterns.append(tidy_input[i][1:])
            continue
        grouped.append([tidy_input[i]])
    if group:
        grouped.append(group)

    print(grouped)

    expanded_paterns: list[list[str]] = []
    for pattern in remove_patterns:
        exp_pattern = parse_simulation(pattern)
        expanded_paterns.extend(exp_pattern)

    result = []
    for group in grouped:
        expanded_groups = []
        for config in group:
            expanded_group = parse_simulation(config)
            expanded_group = filter_simulations(expanded_group, expanded_paterns)

            if not expanded_group:
                print(f'Entire line: "{config}" matches a remove pattern! ' 
                      'Removing it and simulations it would run in paralell with from the pool...')

            expanded_groups.append(expanded_group)
        result.extend(list(itertools.product(*expanded_groups)))

    for configs in result:
        for i,config in enumerate(configs):
            if config[-1].startswith("#"):
                config.remove(config[-1])

    return result