import os

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping


OPTIMIZERS = {
    "Adam" : optim.Adam,
    "SGD" : optim.SGD
}


def score_function(engine):
    """
    Score function for training. Can be modified.
    """
    val_loss = engine.state.metrics['nll']
    return -val_loss


def train_model(model,
                data,
                batch_size=64,
                lr=0.01,
                optimizer=None):
    """
    Train function for models used in nets.py. Used to monitor and apply
    early stopping.
    """
    criterion = nn.NLLLoss()
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.6)
    else:
        optimizer = OPTIMIZERS[optimizer](model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, criterion, device=model.device)

    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=model.device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=model.device)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), log_training_loss)

    size = len(data)
    lengths = [int(size * 0.75), size - int(size * 0.75)]
    trainset, valset = random_split(data,
                                    lengths=lengths,
                                    generator=torch.Generator().manual_seed(42)
                                    )
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    val_loader = DataLoader(valset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4
                            )

    training_history = {'accuracy': [], 'loss': []}
    validation_history = {'accuracy': [], 'loss': []}
    last_epoch = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        accuracy = metrics['accuracy'] * 100
        loss = metrics['nll']
        last_epoch.append(0)
        training_history['accuracy'].append(accuracy)
        training_history['loss'].append(loss)
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, accuracy, loss))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        accuracy = metrics['accuracy'] * 100
        loss = metrics['nll']
        validation_history['accuracy'].append(accuracy)
        validation_history['loss'].append(loss)
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, accuracy, loss))

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer, min_delta=0)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=1000)

    return training_history, validation_history
