from flwr.app import ArrayRecord, MetricRecord, RecordDict
from time import time

import numpy as np
import argparse
import json

def train():
    """Train the model on local data."""
    
    metrics = {
        "train_loss": 0.0,
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)

    arr1 = np.random.randn(3, 3)
    arr2 = np.random.randn(2, 2)
    record = ArrayRecord([arr1, arr2])

    content = RecordDict({"arrays": record, "metrics": metric_record})


def evaluate():
    """Evaluate the model on local data."""

    metrics = {
        "eval_loss": 0.0,
        "eval_acc": 0.0,
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    
def main_exec(config):

    times_epochs = {}

    time_start = time()
    
    train()

    time_end = time()

    times_epochs['fit_1'] = str(time_end-time_start)

    time_start = time()

    evaluate()

    time_end = time()

    times_epochs['eval_1'] = str(time_end - time_start)

    time_start = time()

    train()

    time_end = time()

    times_epochs['fit_2'] = str(time_end - time_start)

    time_start = time()

    evaluate()

    time_end = time()

    times_epochs['eval_2'] = str(time_end - time_start)

    print("times_epochs")
    print(times_epochs)

    with open(config.file, 'w') as f:
        f.write(json.dumps(times_epochs))


if __name__ == "__main__":

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Empty Flower App')

    # Pre Scheduling options
    parser.add_argument('-file', dest='file', type=str, default='times.json', help='File to print execution times')

    config, unparsed = parser.parse_known_args()

    # Run main program
    main_exec(config)