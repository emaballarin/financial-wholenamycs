#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================ #
# Part of:
# "The Stock Transformer: whole-system multidimensional financial time series
# forecasting from timestamped prices via stacked self-attention"
#
# Davide Roznowicz, Emanuele Ballarin <emanuele@ballarin.cc>
#
# (https://github.com/emaballarin/financial-wholenamycs)
# ============================================================================ #

# Imports
import torch as th
from torch.nn import MSELoss, ModuleList, Identity

from data import stock_dataloader_dispatcher
from architectures import StockTransformerModel

from ebtorch.optim import Lookahead, RAdam
from ebtorch.logging import AverageMeter
from ebtorch.nn import Mish

from train_utils import train_epoch

from accelerate import Accelerator

from torchinfo import summary

# DATA:
train_loader, test_loader, _, data_props = stock_dataloader_dispatcher(
    data_path="../data/",
    which_financial=(range(100)),    # <-- Memory-bound
    which_contextual=(0, 1, 2, 3),  # Whole, Year, Month, Whatever
    time_lookback=120,              # Reasonable: 120
    time_predict=5,                 # Almost surely in [5, 10]
    window_stride=1,                # Different from 1 does not make sense!
    ttsr=0.8,                       # Reasonably in [50, 90]
    train_bs=256,                     # <-- Largest possible
    test_bs=512,                    # Default: 512
)

# MODEL PARAMETERS:
A = [
    (
        data_props.fin_size,
        (1, 2),
        (2, 2),
        (3, 2),
        (2, 1),
    ),
    {"batchnorm": True, "activation_fx": Mish()},
]
B = [([200, 400, 80], 16), {"batchnorm": True, "activation_fx": Mish()}]
C = [(220, 2, 2048), {"activation": "gelu", "batch_first": True}]
D = [
    (),
    {"encoder_layer": "_", "num_layers": 10},
]
E = [([12980, 500], 5*100), {"batchnorm": True, "activation_fx": ModuleList((Mish(), Identity()))}]

F = data_props.ctx_size
G = data_props.fin_size

################################################################################

ACCELERATOR: bool = True
AUTODETECT: bool = True
DRY_VALIDATE: bool = False

nrepochs = 2

model = StockTransformerModel(A, B, C, D, E, F, G)

if not ACCELERATOR:
    device = th.device("cuda" if th.cuda.is_available() and AUTODETECT else "cpu")
    model = model.to(device)
    accelerator = None
else:
    device = None
    accelerator = Accelerator()

criterion = MSELoss(reduction="mean")
optimizer = RAdam(model.parameters())   # rl, mom?, betas??
train_acc_avgmeter = AverageMeter("Training Loss")

base_optimizer = optimizer
optimizer = Lookahead(base_optimizer)   # la_steps? (e.g. 5->4)

# scheduler = MultiStepLR(optimizer, milestones=[10, 11], gamma=0.4)

if ACCELERATOR:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

################################################################################

if DRY_VALIDATE:
    for _, dry_data in enumerate(train_loader):
        dry_x, dry_y_ = dry_data
        dry_y = th.flatten(dry_y_, start_dim=2, end_dim=3).transpose(-1, -2)
        break

    print("VALIDATING...")
    print(summary(model, input_data=dry_x))

else:
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

        # # Tweaks for the Lookahead optimizer (before testing)
        # if isinstance(optimizer, Lookahead):
        #     optimizer._backup_and_load_cache()

        # Testing: on training and testing set
        #    print("TESTING...")
        #    print("\nON TRAINING SET:")
        #    _ = test(model, device, test_on_train_loader, lossfn, quiet=False)
        #    print("\nON TEST SET:")
        #    _ = test(model, device, test_loader, lossfn, quiet=False)
        #    print("\n\n")

        # # Tweaks for the Lookahead optimizer (after testing)
        # if isinstance(optimizer, Lookahead):
        #     optimizer._clear_and_load_backup()

        # Scheduling step (outer)
        # scheduler.step()
