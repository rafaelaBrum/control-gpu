from typing import Dict

import argparse
import numpy as np
import os
import flwr as fl

import writer_daemon as metrics_writer

from fedavg_strategy import FedAvg
from fedavg_saving_strategy import FedAvgSave

DEFAULT_SERVER_ADDRESS = "[::]:8080"

strategy_print = "Strategy needs to be one of the following:\n" \
                 "--> FedAvg\n" \
                 "--> FedAvgSave\n"


def get_args():
    parser = argparse.ArgumentParser(description="Creating Flower server")
    parser.add_argument(
        "--server_address", type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds", type=int, required=True,
        help="Number of rounds of federated learning",
        default=20
    )
    parser.add_argument(
        "--sample_fraction", type=float, required=True,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size", type=int, required=True,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients", type=int, required=True,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--strategy", type=str,
        default='FedAvg',
        help="Type of used strategy",
    )
    parser.add_argument(
        "--log_host", type=str,
        help="Log server address (no default)",
    )

    parser.add_argument('--frequency_ckpt', help='Frequency of checkpointing when using FedAvgSave', type=int,
                        default=None)

    parser.add_argument("--ckpt_restore", action="store_true", dest="ckpt_restore", default=False)
    args = parser.parse_args()
    return args


def get_initial_weights(args):
    try:
        if args.ckpt_restore:
            while not os.path.exists('weights.npz'):
                continue
        file = 'weights.npz'
        aux_data = np.load(file)
        aux_list: fl.common.NDArrays = []
        for file in aux_data.files:
            aux_list.append(aux_data[file])
        weights = fl.common.ndarrays_to_parameters(aux_list)
        return weights
    except Exception:
        return None


def main():
    """Start server and train five rounds."""
    args = get_args()

    # Create strategy
    strategy = None

    initial_weights = get_initial_weights(args)

    if initial_weights is not None:
        args.strategy = "FedAvgSave"

    print("strategy:", args.strategy)

    if args.strategy.upper() == "FEDAVG":
        strategy = FedAvg(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config
        )
    elif args.strategy.upper() == "FEDAVGSAVE":
        if initial_weights is not None:
            strategy = FedAvgSave(
                fraction_fit=args.sample_fraction,
                fraction_eval=args.sample_fraction,
                min_fit_clients=args.min_sample_size,
                min_eval_clients=args.min_sample_size,
                min_available_clients=args.min_num_clients,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                initial_parameters=initial_weights,
                frequency_ckpt=args.frequency_ckpt
            )
        else:
            strategy = FedAvgSave(
                fraction_fit=args.sample_fraction,
                fraction_eval=args.sample_fraction,
                min_fit_clients=args.min_sample_size,
                min_eval_clients=args.min_sample_size,
                min_available_clients=args.min_num_clients,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                frequency_ckpt=args.frequency_ckpt
            )
    else:
        print(strategy_print)
        exit(1)

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    # fl.server.start_server("[::]:8080", config={"num_rounds": 3})
    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )


def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config: Dict[str, fl.common.Scalar] = {
        "epoch_global": str(rnd),
    }
    return config


if __name__ == "__main__":
    main()
