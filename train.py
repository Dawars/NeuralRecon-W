import os
import math
from opt import get_opts

from datasets import DataModule
from lightning_modules.neuconw_system import NeuconWSystem

# pytorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from config.defaults import get_cfg_defaults


def main(hparams):

    config = get_cfg_defaults()
    config.merge_from_file(hparams.cfg_path)

    caches = None
    pl.seed_everything(config.TRAINER.SEED)

    # scale lr and warmup-step automatically
    config.TRAINER.WORLD_SIZE = hparams.num_gpus * hparams.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * hparams.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.LR = config.TRAINER.CANONICAL_LR * _scaling

    if hasattr(hparams, 'shadow_weight'):
        config.NEUCONW.LOSS.shadow_weight = hparams.shadow_weight

    system = NeuconWSystem(hparams, config, caches) 

    data_module = DataModule(hparams, config)
    
    os.makedirs(hparams.save_path, exist_ok=True)
    os.makedirs(os.path.join(hparams.save_path, hparams.exp_name, 'ckpts'), exist_ok=True)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams.save_path, hparams.exp_name, 'ckpts'),
                        filename='{epoch:d}',
                        monitor='val/psnr',
                        mode='max',
                        every_n_train_steps=config.TRAINER.SAVE_FREQ,
                        save_top_k=-1)

    logger = TensorBoardLogger(save_dir=hparams.save_path,
                               name=hparams.exp_name,
                               log_graph=False)
    if config.DATASET.DATASET_NAME == 'phototourism' and config.DATASET.PHOTOTOURISM.IMG_DOWNSCALE <= 1:
        replace_sampler_ddp = False
    else:
        replace_sampler_ddp = True


    profiler = "simple" if hparams.num_gpus == 1 else None
    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback, ],  # DeviceStatsMonitor(cpu_stats=True)],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      devices=hparams.num_gpus,
                      num_nodes=hparams.num_nodes,
                      accelerator='cuda',
                      strategy='ddp' if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
                      val_check_interval=config.TRAINER.VAL_FREQ,
                      benchmark=True,
                      profiler=profiler,
                      replace_sampler_ddp=replace_sampler_ddp,  # need to read all data of local dataset when config.DATASET.PHOTOTOURISM.IMG_DOWNSCALE==1
                      gradient_clip_val=0.99
                      )

    trainer.fit(system, datamodule=data_module)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
