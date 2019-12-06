[pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning) version of Unet.


## Train

+ Put your dataset in `dataset/{dataset_name}`:
    + `train`: contains image names
    + `train_masks`: contains image masks

+ Sample with `carvana` dataset:

```
python train.py --dataset carvana --n_channels 3
```

Log and checkpoints are saved automatically in `lightning_logs`, thank to pytorch-lightning.


## Test

+ Sample with `carvana` dataset:

```
python test.py --checkpoint lightning_logs/version_0/checkpoints/_ckpt_epoch_1.ckpt --img_dir dataset/carvana/test --out_dir result/carvana
```

## Reference

+ Implementation is heavily borrowed from [milesial](https://github.com/milesial/Pytorch-UNet)
