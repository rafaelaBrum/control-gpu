from typing import Dict

import flwr as fl
import argparse
import numpy as np
import pickle

from google.cloud import storage

from fedavg_strategy import FedAvg
from fast_and_slow_strategy import FastAndSlow
from ft_fedavg_strategy import FaultTolerantFedAvg
from fedadagrad_strategy import FedAdagrad
from fedadam_strategy import FedAdam
from fedavg_m_strategy import FedAvgM
from fedfs_v0_strategy import FedFSv0
from fedfs_v1_strategy import FedFSv1
from fedopt_strategy import FedOpt
from fedyogi_strategy import FedYogi
from qfedavg_strategy import QFedAvg
from fedavg_saving_strategy import FedAvgSave


DEFAULT_SERVER_ADDRESS = "[::]:8080"

strategy_print = "Strategy needs to be one of the following:\n" \
                 "--> FedAvg\n" \
                 "--> Fast_Slow\n" \
                 "--> FTFedAvg\n" \
                 "--> FedAdagrad\n" \
                 "--> FedAdam\n" \
                 "--> FedAvgM\n" \
                 "--> FedFSv0\n" \
                 "--> FedFSv1\n" \
                 "--> FedOpt\n" \
                 "--> FedYogi\n" \
                 "--> QFedAvg\n" \
                 "--> FedAvgSave\n"


def download_blob(file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    bucket_name = "fl_server_weights"

    # The ID of your GCS object
    source_blob_name = "vgg16_weights.bin"

    # The path to which the file should be downloaded
    destination_file_name = file_name

    print(
        "Start to download storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


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
    parser.add_argument(
        "--file_weights", type=str,
        default='vgg16_weights.bin',
        help="File where initial weights are stored (default: weights.bin)",
    )

    parser.add_argument('--frequency_ckpt', help='Frequency of checkpointing when using FedAvgSave', type=int,
                        default=None)
    args = parser.parse_args()
    return args


def get_initial_weights():
    try:
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

    initial_weights = get_initial_weights()

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
    elif args.strategy.upper() == "FAST_SLOW":
        strategy = FastAndSlow(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config
        )
    elif args.strategy.upper() == "FTFEDAVG":
        strategy = FaultTolerantFedAvg(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config
        )
    elif args.strategy.upper() == "FEDADAGRAD":
        download_blob(args.file_weights)
        # Read list to memory
        # for reading also binary mode is important
        with open(args.file_weights, 'rb') as fp:
            initial_weights = pickle.load(fp)
        initial_parameters = fl.common.weights_to_parameters(initial_weights)
        strategy = FedAdagrad(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            initial_parameters=initial_parameters
        )
    elif args.strategy.upper() == "FEDADAM":
        download_blob(args.file_weights)
        # Read list to memory
        # for reading also binary mode is important
        with open(args.file_weights, 'rb') as fp:
            initial_weights = pickle.load(fp)
        initial_parameters = fl.common.weights_to_parameters(initial_weights)
        strategy = FedAdam(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            initial_parameters=initial_parameters
        )
    elif args.strategy.upper() == "FEDAVGM":
        strategy = FedAvgM(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config
        )
    elif args.strategy.upper() == "FEDFSV0":
        strategy = FedFSv0(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config
        )
    elif args.strategy.upper() == "FEDFSV1":
        strategy = FedFSv1(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config
        )
    elif args.strategy.upper() == "FEDOPT":
        download_blob(args.file_weights)
        # Read list to memory
        # for reading also binary mode is important
        with open(args.file_weights, 'rb') as fp:
            initial_weights = pickle.load(fp)
        initial_parameters = fl.common.weights_to_parameters(initial_weights)
        strategy = FedOpt(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            initial_parameters=initial_parameters
        )
    elif args.strategy.upper() == "FEDYOGI":
        download_blob(args.file_weights)
        # Read list to memory
        # for reading also binary mode is important
        with open(args.file_weights, 'rb') as fp:
            initial_weights = pickle.load(fp)
        initial_parameters = fl.common.weights_to_parameters(initial_weights)
        strategy = FedYogi(
            fraction_fit=args.sample_fraction,
            fraction_eval=args.sample_fraction,
            min_fit_clients=args.min_sample_size,
            min_eval_clients=args.min_sample_size,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            initial_parameters=initial_parameters
        )
    elif args.strategy.upper() == "QFEDAVG":
        strategy = QFedAvg(
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
        # "epochs": str(10),
    }
    return config


if __name__ == "__main__":
    main()
