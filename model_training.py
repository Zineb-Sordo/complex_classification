import argparse
import os
import pytorch_lightning as pl
import torch

import argparse
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pathlib import Path
import fire

from model_setup import RSS
from data_processing import KneeDataModule


# get the data
def get_data(args: argparse.ArgumentParser) -> pl.LightningDataModule:
    # get datamodule
    if args.data_type == "knee":
        # load mc data to obtain rss images
        datamodule = KneeDataModule(
            label_type="knee",
            split_csv_file=args.split_csv_file,
            coil_type=args.coil_type,
            batch_size=args.batch_size,
            sampler_filename=args.sampler_filename,
            data_space=args.data_space,
        )
    else:
        raise NotImplementedError

    return datamodule


def get_model(
    args: argparse.ArgumentParser, device: torch.device,
) -> pl.LightningModule:
    if args.data_type == "knee":
        # get spatial domain model
        model = RSS(
            model_type=args.model_type,
            data_type=args.data_type,
            image_shape=[320, 320],
            drop_prob=args.drop_prob,
            kspace_shape=[640, 400],
            data_space=args.data_space,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_gamma=args.lr_gamma,
            lr_step_size=args.lr_step_size,
        )
    else:
        raise NotImplementedError
    return model


def train_model(
        args: argparse.Namespace,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        device: torch.device,
) -> pl.LightningModule:
    log_dir = (
            Path(args.log_dir)
            / args.data_type
            / args.data_space
    )
    model_dir = str(args.model_dir) + '/' + args.data_space + '/' + str(args.n_seed)

    if not os.path.isdir(str(log_dir)):
        os.makedirs(str(log_dir))
    if not os.path.isdir(str(model_dir)):
        os.makedirs(str(model_dir))

    csv_logger = CSVLogger(save_dir=log_dir, name=f"train-{args.n_seed}", version=f"{args.n_seed}")
    wandb_logger = WandbLogger(name=f"{args.data_space}-{args.n_seed}")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(monitor='val_auc_mean', dirpath=model_dir, filename="{epoch:02d}-{val_auc_mean:.2f}" ,save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor='val_auc_mean', patience=5, mode='max')
    print("In train_model tune and {}".format(str(device).startswith("cuda")))

    trainer: pl.Trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.n_devices,
        max_epochs=args.n_epochs,
        replace_sampler_ddp=False,
        logger=[wandb_logger, csv_logger],
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
        auto_lr_find=True,
    )

    # trainer: pl.Trainer = pl.Trainer(
    #     gpus=1 if str(device).startswith("cuda") else 0,
    #     max_epochs=args.n_epochs,
    #     # logger=wandb_logger,
    #     logger=csv_logger,
    #     # logger=[wandb_logger, csv_logger],
    #     callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
    #     auto_lr_find=True,
    # )
    # Runs a learning rate finder algorithm when calling trainer.tune() to find optimate lr

    trainer.tune(model, datamodule)
    print("In train_model fit and {}".format(str(device).startswith("cuda")))
    trainer: pl.Trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.n_devices,
        strategy=args.strategy,
        max_epochs=args.n_epochs,
        replace_sampler_ddp=False,
        logger=[wandb_logger, csv_logger],
        #logger=wandb_logger,
        #logger=csv_logger,
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
    )

    # trainer: pl.Trainer = pl.Trainer(
    #     gpus=1 if str(device).startswith("cuda") else 0,
    #     max_epochs=args.n_epochs,
    #     #logger=[wandb_logger, csv_logger],
    #     # logger=wandb_logger,
    #     logger=csv_logger,
    #     callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
    # )
    trainer.fit(model, datamodule)
    print("Finished training model")
    return model


def test_model(
        args: argparse.Namespace,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        device: torch.device,
) -> pl.LightningModule:

    model_dir = str(args.model_dir) + '/' + args.data_space + '/'  + str(args.n_seed)
    checkpoint_filename = os.listdir(model_dir)[0]
    print("Checkpoint file: ", model_dir, checkpoint_filename)
    log_dir = (
            Path(args.log_dir)
            / args.data_type
            / args.data_space
    )

    csv_logger = CSVLogger(save_dir=log_dir, name=f"test-{args.n_seed}", version=f"{args.n_seed}")
    model = RSS.load_from_checkpoint(model_dir + '/' + checkpoint_filename)
    trainer = pl.Trainer(logger=csv_logger,
        accelerator=args.accelerator,
        devices=args.n_devices,
        strategy=args.strategy,)
    #trainer = pl.Trainer(logger=csv_logger, gpus=1 if str(device).startswith("cuda") else 0)

    with torch.inference_mode():
        model.eval()
        M_val = trainer.validate(model, datamodule.val_dataloader())
        M = trainer.test(model, datamodule.test_dataloader())

    print("Finish testing")


def run_experiment(args):

    print(args, flush=True)

    if torch.cuda.is_available():
        print("Found CUDA device, running job on GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = get_data(args)
    model = get_model(args, device)
    if args.mode == "train":
        model = train_model(args=args, model=model, datamodule=datamodule, device=device,)
    else:
        datamodule.setup()
        test_model(args=args, model=model, datamodule=datamodule, device=device,)


def get_args():

    parser = argparse.ArgumentParser(description="Indirect MR Screener training")

    # logging parameters
    parser.add_argument("--model_dir", type=str, default="./trained_models")
    parser.add_argument("--log_dir", type=str, default="./trained_logs")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dev_mode", action="store_true")

    # data parameters
    parser.add_argument("--data_type", type=str, default="knee",)
    parser.add_argument("--data_space", type=str, default="complex_input")
    # parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--image_shape", type=int, default=[320, 320], nargs=2, required=False)
    parser.add_argument("--image_type", type=str, default='orig', required=False, choices=["orig"])

    # parser.add_argument("--split_csv_file", type=str, default='..//metadata_knee.csv', required=False)
    parser.add_argument("--split_csv_file",
                        type=str,
                        default='./knee/metadata_knee.csv',
                        required=False)
    parser.add_argument("--recon_model_ckpt", type=str)
    parser.add_argument("--recon_model_type", type=str, default=["rss"], required=False, choices=["rss"])
    parser.add_argument("--mask_type", type=str, default="none")
    parser.add_argument("--k_fraction", type=float, default=0.25)
    parser.add_argument("--center_fraction", type=float, default=0.08)
    parser.add_argument("--coil_type", type=str, default="sc", choices=["sc", "mc"])

    parser.add_argument("--sampler_filename", type=str, default="./sampler_knee_tr.p")
    parser.add_argument(
        "--model_type",
        type=str,
        default="complex_preact_resnet18",
        choices=["complex_preact_resnet18",
                 "complex_preact_resnet50"
                 ],
    )

    # training parameters
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="dp")
    parser.add_argument("--accelerator", type=str, default='cpu')
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=5)

    parser.add_argument("--n_masks", type=int, default=100)

    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--sweep_step", type=int)
    parser.add_argument('--debug',  default=True)

    args, unkown = parser.parse_known_args()

    return args


def main(sweep_step=None):
    args = get_args()
    run_experiment(args)


if __name__ == "__main__":
    fire.Fire(main)
