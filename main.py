import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger


def main(hparams):
    model = Unet(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_best_only=False,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix='unet-'
    )
    logger = TestTubeLogger(
        save_dir='./lightning_logs',
        version=1,
    )
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
