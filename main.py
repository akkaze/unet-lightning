import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer


def main(hparams):
    model = Unet(hparams)

    trainer = Trainer(
        gpus=1,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()
    main(hparams)
