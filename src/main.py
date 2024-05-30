from datetime import datetime
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse
import os
from script_utils import load_data, load_qm9_data
from model.pp_bridge import PPBridge


def main(args):
    # Load the configuration
    config = OmegaConf.load(args.config)
    log_name = args.config.split('/')[-1].split('.')[0]

    dataset_root_path = config.data.root # '/data/conghao001/pharmacophore2drug/PP2Drug/data/small_dataset' # config.data.root
    print(f'Loading data from {dataset_root_path}')
    # train_dataset, train_loader = load_data(dataset_root_path, split='all', batch_size=config.training.batch_size)

    datamodule = config.data.module
    if datamodule == 'QM9Dataset':
        _, _, train_loader, val_loader = load_qm9_data(root=dataset_root_path, split='train', batch_size=config.training.batch_size)
    else:
        _, train_loader = load_data(datamodule, dataset_root_path, split='train', batch_size=config.training.batch_size)
        _, val_loader = load_data(datamodule, dataset_root_path, split='valid', batch_size=config.training.batch_size)
        # test_dataset, test_loader = load_data(dataset_root_path, split='test', batch_size=config.training.batch_size)

    model = PPBridge(config)
    now = str(datetime.now()).replace(" ", "_").replace(":", "_")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)    # should be the ema loss
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=f"lightning_logs/{log_name}_{datamodule}_{now}",
        mode="min",
        filename='epoch={epoch:02d}-val_loss={val_loss:.2f}',
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        # max_steps=1,
        devices=[int(args.gpu)],
        logger=WandbLogger(project='PP2Drug', name=log_name, log_model='all'),
        callbacks=[lr_monitor, early_stopping, checkpoint_callback],
        log_every_n_steps=1,
        # progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,  # Clip the gradient norm to 0.5
        gradient_clip_algorithm='norm',
    )
    trainer.fit(model, train_loader, val_loader)

    with open(os.path.join(f'lightning_logs/{log_name}_{datamodule}_{now}', args.config.split('/')[-1]), 'w') as f:
        OmegaConf.save(config=config, f=f.name)     # trainer.log_dir is the path of main.py, not the log path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/vp_bridge.yml', help='Path to the configuration file')
    parser.add_argument('--gpu', '-g', type=int, default=3, help='Which GPU to use')
    args = parser.parse_args()
    main(args)