#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================ #
# Part of:
# "Whole-system multidimensional financial time series prediction and simulation
# from timestamped prices only"
#
# Davide Roznowicz, Emanuele Ballarin <emanuele@ballarin.cc>
#
# (https://github.com/emaballarin/financial-wholenamycs)
# ============================================================================ #

# Imports
import torch as th
from torch.nn import MSELoss

from data import stock_dataloader_dispatcher
from architectures import StockTransformerModel

from ebtorch.optim import Lookahead, RAdam
from ebtorch.logging import AverageMeter

from train_utils import train_epoch

from accelerate import Accelerator

# DATA:
train_loader, test_loader, _, data_props = stock_dataloader_dispatcher(
    data_path="../data/",
    which_financial=(0, 1, 2),
    which_contextual=(0, 1, 2, 3, 4),
    time_lookback=20,
    time_predict=5,
    window_stride=1,
    ttsr=0.8,
    train_bs=2,
    test_bs=512,
)

# MODEL PARAMETERS:
A = [
    (
        data_props.fin_size,
        [1, 2],
        [2, 4],
        [2, 2],
        [3, 1],
    ),
    {"batchnorm": [True, True]},
]
B = [([12, 2, 2], 3)]
C = [(20, 2, 10), {"batch_first": True}]
D = [
    (),
    {"encoder_layer": "_", "num_layers": 2},
]
E = [([20 * 6, 10], 15)]

F = data_props.ctx_size
G = data_props.fin_size

# MODEL:


################################################################################

nrepochs = 2
ACCELERATOR: bool = False
AUTODETECT: bool = True

model = StockTransformerModel(A, B, C, D, E, F, G)

if not ACCELERATOR:
    device = th.device("cuda" if th.cuda.is_available() and AUTODETECT else "cpu")
    model = model.to(device)
    accelerator = None
else:
    device = None
    accelerator = Accelerator()

criterion = MSELoss(reduce=True)
optimizer = RAdam(model.parameters())
train_acc_avgmeter = AverageMeter("Training Loss")
# base_optimizer = MadGrad(model.parameters(), lr=0.00017)
# optimizer = Lookahead(base_optimizer, la_steps=4)
# scheduler = MultiStepLR(optimizer, milestones=[10, 11], gamma=0.4)

if ACCELERATOR:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

################################################################################

for epoch in range(1, nrepochs + 1):

    # Training
    print("TRAINING...")

    train_epoch(
        model=model,
        device=device,
        train_loader=train_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        epoch=epoch,
        print_every_nep=1,
        train_acc_avgmeter=train_acc_avgmeter,
        inner_scheduler=None,
        accelerator=accelerator,
        quiet=False,
    )

    # Tweaks for the Lookahead optimizer (before testing)
    if isinstance(optimizer, Lookahead):
        optimizer._backup_and_load_cache()

    # Testing: on training and testing set
    #    print("\nON TRAINING SET:")
    #    _ = test(model, device, test_on_train_loader, lossfn, quiet=False)
    #    print("\nON TEST SET:")
    #    _ = test(model, device, test_loader, lossfn, quiet=False)
    #    print("\n\n")

    # Tweaks for the Lookahead optimizer (after testing)
    if isinstance(optimizer, Lookahead):
        optimizer._clear_and_load_backup()

    # Scheduling step (outer)
    # scheduler.step()
