from flwr.app import ArrayRecord, MetricRecord, RecordDict
from time import time

import numpy as np
import argparse
import json

import torch
import torch.nn as nn
import timm

from typing import OrderedDict

from torch.utils.data import DataLoader
import torch.optim as optim

from torchmetrics import F1Score, Precision, Recall, AUROC

from datasets import load_dataset, load_from_disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vit_base(num_classes: int, pretrained=True) -> nn.Module:

    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model.to(DEVICE)

def _zero_like_control_variates(model) -> OrderedDict[str, torch.Tensor]:
    control_dict = OrderedDict()
    for name, param in model.named_parameters():
        control_dict[name] = torch.zeros_like(param)
    return control_dict


def _get_param_state(model) -> OrderedDict[str, torch.Tensor]:
    # Return a detached clone of parameter tensors only (exclude buffers)
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def train_local(model, loader, epochs, lr, c_global=None, c_client=None):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    losses, accs = [], []

    use_scaffold = c_global is not None and c_client is not None

    num_steps = 0
    for _ in range(epochs):
        tot_loss, corr, tot = 0.0, 0, 0
        for batch in loader:
            imgs  = batch["pixel_values"].to(DEVICE)
            lbls  = batch["labels"].to(DEVICE)
            opt.zero_grad()
            out   = model(imgs)
            loss  = crit(out, lbls)
            loss.backward()

            if use_scaffold:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        param.grad.add_(c_global[name] - c_client[name])

            opt.step()

            tot_loss += loss.item() * lbls.size(0)
            preds    = out.argmax(dim=1)
            corr    += (preds == lbls).sum().item()
            tot     += lbls.size(0)
            num_steps += 1

        losses.append(tot_loss / tot)
        accs.append(100 * corr / tot)
    return model.state_dict(), losses, accs, num_steps


def train(num_classes, local_epochs, lr, batch_size):
    """Train the model on local data."""

    local_model = vit_base(num_classes, False)

    global_c = _zero_like_control_variates(local_model)
    client_c = global_c.copy()


    start_params = _get_param_state(local_model)
    total_local_steps = 0

    train_loader = DataLoader(load_from_disk(f"data/train"), batch_size=batch_size, shuffle=True)

    # train one epoch at a time
    for _ in range(local_epochs):
        # train for exactly 1 epoch
        state_dict, [train_loss], [train_acc], steps = train_local(
            local_model,
            train_loader,
            epochs=1,   # one epoch per call
            lr=lr,
            c_global=global_c,
            c_client=client_c,
        )
        total_local_steps += steps

    end_params = _get_param_state(local_model)

    delta_c: OrderedDict[str, torch.Tensor] = {}

    if total_local_steps > 0:
        inv_lr_T = 1.0 / (lr * float(total_local_steps))
        new_client_c: OrderedDict[str, torch.Tensor] = {}
        for name in client_c.keys():
            # c_i' = c_i - c + (w_start - w_end)/(eta * T)
            ci_prime = client_c[name] - global_c[name] + (start_params[name] - end_params[name]) * inv_lr_T
            new_client_c[name] = ci_prime
            delta_c[name] = ci_prime - client_c[name]
    else:
        # no steps -> no update to c_i
        delta_c = {name: torch.zeros_like(t) for name, t in client_c.items()}

    metrics = {
        "train_loss": train_loss,
        "train-acc": train_acc,
        "num-examples": len(train_loader.dataset),
    }

    metric_record = MetricRecord(metrics)

    record = ArrayRecord(local_model.state_dict())

    delta_c_record = ArrayRecord(delta_c)

    content = RecordDict({"arrays": record, "global-c": delta_c_record, "metrics": metric_record})


def evaluate_local(model, loader, num_classes):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss, corr, tot = 0.0, 0, 0
    
    # Initialize metrics
    f1 = F1Score(task='multiclass', num_classes=num_classes).to(DEVICE)
    precision = Precision(task='multiclass', num_classes=num_classes).to(DEVICE)
    recall = Recall(task='multiclass', num_classes=num_classes).to(DEVICE)
    auroc = AUROC(task='multiclass', num_classes=num_classes).to(DEVICE)
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for batch in loader:
            imgs  = batch["pixel_values"].to(DEVICE)
            lbls  = batch["labels"].to(DEVICE)
            out   = model(imgs)
            probs = torch.softmax(out, dim=1)
            
            tot_loss += crit(out, lbls).item() * lbls.size(0)
            preds    = out.argmax(dim=1)
            corr    += (preds == lbls).sum().item()
            tot     += lbls.size(0)
            
            all_preds.append(preds)
            all_labels.append(lbls)
            all_probs.append(probs)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Calculate metrics
    f1_score = f1(all_preds, all_labels).item()
    prec = precision(all_preds, all_labels).item()
    rec = recall(all_preds, all_labels).item()
    auc = auroc(all_probs, all_labels).item()
    
    return {
        'loss': tot_loss / tot,
        'accuracy': 100 * corr / tot,
        'f1_score': f1_score,
        'precision': prec,
        'recall': rec,
        'auc': auc
    }


def evaluate(num_classes, batch_size):
    """Train the model on local data."""
    local_model = vit_base(num_classes)

    test_loader = DataLoader(load_from_disk(f"data/test"), batch_size=batch_size, shuffle=False)


    """Evaluate the model on local data."""
    metrics = evaluate_local(model=local_model, loader=test_loader, num_classes=num_classes)
    metrics['num-examples'] = len(test_loader.dataset)
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    
def main_exec(config):

    times_epochs = {}

    time_start = time()
    
    train(config.num_classes, config.local_epochs, config.lr, config.batch_size)

    time_end = time()

    times_epochs['fit_1'] = str(time_end-time_start)

    time_start = time()

    evaluate(config.num_classes, config.batch_size)

    time_end = time()

    times_epochs['eval_1'] = str(time_end - time_start)

    time_start = time()

    train(config.num_classes, config.local_epochs, config.lr, config.batch_size)

    time_end = time()

    times_epochs['fit_2'] = str(time_end - time_start)

    time_start = time()

    evaluate(config.num_classes, config.batch_size)

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

    parser.add_argument('-num_classes', dest='num_classes', type=int, default=8, help='Number of classes')

    parser.add_argument('-local-epochs', dest='local_epochs', type=int, default=5, help='Quantity of local epochs in training')

    parser.add_argument('-lr', dest='lr', type=float, default=0.01, help="Learning rate in training")

    parser.add_argument('-batch-size', dest='batch_size', type=int, default=16, help="Size of batch to be used in training")

    config, unparsed = parser.parse_known_args()

    print(config)

    # Run main program
    main_exec(config)