from typing import Dict

import flwr as fl
import argparse

from fedavg_strategy import FedAvg

DEFAULT_SERVER_ADDRESS = "[::]:8080"


def get_args():
    parser = argparse.ArgumentParser(description="Testando criar o servidor para FCUBE automaticamente")
    parser.add_argument(
        "--server_address", type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds", type=int, required=True,
        help="Number of rounds of federated learning",
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
        "--log_host", type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()
    return args


def main():
    """Start server and train five rounds."""
    args = get_args()

    # Create strategy
    # strategy = fl.server.server.FedAvg(
    strategy = FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_eval=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_eval_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        on_fit_config_fn=fit_config
    )

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
        "epochs": str(1),
    }
    return config


if __name__ == "__main__":
    main()
