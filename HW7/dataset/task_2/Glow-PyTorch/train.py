import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from dataset import CelebALoader
from model import Glow
from utils import save_image
import wandb


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses



def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}
    y_logits = torch.sigmoid(y_logits)
    y = y.float()
    loss_classes = F.binary_cross_entropy_with_logits(y_logits, y)
    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    # wandb.log({"total_loss": losses["total_loss"].item(), "loss_classes": losses["loss_classes"].item()})

    return losses


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)

def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))

def main(
    dataset,
    dataroot,
    download,
    augment,
    batch_size,
    eval_batch_size,
    epochs,
    saved_model,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    n_workers,
    cuda,
    n_init_batches,
    output_dir,
    saved_optimizer,
    warmup,
    classifier_weight
):

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    # wandb.init(project=args.dataset)

    check_manual_seed(seed)

    image_shape = (64,64,3)
    num_classes = 40

    # Note: unsupported for now
    multi_class = True #It's True but this variable doesn't be used now


    dataset_train = CelebALoader() 
    train_loader = DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,drop_last=True)

    model = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )

    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # wandb.watch(model)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            ### x: torch.Size([batchsize, 3, 64, 64]); y: torch.Size([batchsize, 24]); z: torch.Size([batchsize, 48, 8, 8])
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)

        losses["total_loss"].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()


        return losses


    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "glow", n_saved=None, require_empty=False
    )
    ### n_saved (Optional[int]) – Number of objects that should be kept on disk. Older files will be removed. If set to None, all objects are kept.

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        {"model": model, "optimizer": optimizer},
    )

    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )


    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)


    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location="cpu")['model'])
        model.set_actnorm_init()

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)

    trainer.run(train_loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="task2",
        choices=["task1", "task2"],
        help="Type of the dataset to be used.",
    )

    parser.add_argument("--dataroot", type=str, default="./", help="path to dataset")

    parser.add_argument("--download", action="store_true", help="downloads dataset")

    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Augment training data",
    )

    parser.add_argument(
        "--hidden_channels", type=int, default=512, help="Number of hidden channels"
    )

    parser.add_argument("--K", type=int, default=32, help="Number of layers per block")

    parser.add_argument("--L", type=int, default=3, help="Number of blocks")

    parser.add_argument(
        "--actnorm_scale", type=float, default=1.0, help="Act norm scale"
    )

    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )

    parser.add_argument(
        "--no_LU_decomposed",
        action="store_false",
        dest="LU_decomposed",
        help="Train with LU decomposed 1x1 convs",
    )

    parser.add_argument(
        "--no_learn_top",
        action="store_false",
        help="Do not train top layer (prior)",
        dest="learn_top",
    )

    parser.add_argument(
        "--y_condition", action="store_true", help="Train using class condition"
    )

    parser.add_argument(
        "--y_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--max_grad_clip",
        type=float,
        default=0,
        help="Max gradient value (clip above - for off)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Max norm of gradient (clip above - 0 for off)",
    )

    parser.add_argument(
        "--n_workers", type=int, default=6, help="number of data loading workers"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size used during training"
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="batch size used during evaluation",
    )

    parser.add_argument(
        "--epochs", type=int, default=250, help="number of epochs to train for"
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    parser.add_argument(
        "--warmup",
        type=float,
        default=5,
        help="Use this number of epochs to warmup learning rate linearly from zero to learning rate",  # noqa
    )

    parser.add_argument(
        "--n_init_batches",
        type=int,
        default=8,
        help="Number of batches to use for Act Norm initialisation",
    )

    parser.add_argument(
        "--no_cuda", action="store_false", dest="cuda", help="Disables cuda"
    )

    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to output logs and model checkpoints",
    )

    parser.add_argument(
        "--fresh", action="store_true", help="Remove output directory before starting"
    )

    parser.add_argument(
        "--saved_model",
        default="",
        help="Path to model to load for continuing training",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    parser.add_argument("--seed", type=int, default=0, help="manual seed")

    parser.add_argument(
        "--classifier_weight",
        default="",
        help="full path of classifier_weight for task1",
    )

    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (
            len(os.listdir(args.output_dir)) > 0
        ):
            raise FileExistsError(
                "Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag."  # noqa
            )

    kwargs = vars(args)
    del kwargs["fresh"]

    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    main(**kwargs)
