from functools import wraps, partial
from dask.distributed import Client, wait, progress


## decorator factory for simulation
def simulate(parameter_range, n_sim, client=None):
    """Decorator factory for simulating a function over a range of parameters. Use as a decorator for a function which takes keyword argument "param"

    Parameters
    ----------
    parameter_range (list-like): sequence of parameters for which to run simulations
    n_sim (int): number of simulations to run for each parameter
    client (dask.distributed.Client): Optional, dask client for parallel computing

    Returns
    -------
    If Dask client is provided or defined in global context:
        futures (list[dask.distributed.Future]): list of Dask futures corresponding to individual simulations
        gather (partial function): function to gather futures into a structured dictionary of results, nested by parameter
    Otherwise:
        result (dict): structured dictionary of results, nested by parameter

    Notes
    -----
    Best use case is to run in a Jupyter notebook with a dask client instantiated in an earlier cell.
    """

    def sim_decorator(function):
        ## decorator that will replace the function it wraps
        wraps(function)
        print(f"Running {n_sim} simulations")
        try:
            nonlocal client
            ## grab client from global env if present, otherwise will raise ValueError
            if client is None:
                client = Client.current()
            print(f"Using dask client at {client.dashboard_link}")

            def wrapper(*args, **kwargs):
                futures = []
                for i in range(n_sim):
                    for p in parameter_range:
                        futures.append(
                            client.submit(
                                function,
                                *args,
                                param=p,
                                seed=i,
                                simno=i,
                                retries=1,
                                **kwargs,
                            )
                        )
                print(f"{len(futures)} parallel jobs")

                ## helper to properly gather and sort distributed jobs
                def gather(parameter_range, futures):
                    n_params = len(parameter_range)
                    gathered_futures = [
                        f.result() if f.status == "finished" else None for f in futures
                    ]
                    result = {p: {} for p in parameter_range}
                    for i in range(len(futures)):
                        result[parameter_range[i % n_params]][i // n_params] = (
                            gathered_futures[i]
                        )
                    return result

                ## return the futures, and a gathering function to be run when the jobs finish
                return futures, partial(gather, parameter_range)

        ## if no client is available, run sequentially
        except ValueError as e:
            print("No dask client available, running sequentially")

            def wrapper(*args, **kwargs):
                result = {p: {} for p in parameter_range}
                for i in range(n_sim):
                    for p in parameter_range:
                        result[p][i] = function(*args, param=p, **kwargs)
                return result

        return wrapper

    return sim_decorator
