import os
import time
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from DatasetModule import TripAdvisorDataModule
from ModelModule import TripAdvisorModelModule, BertTextCNN, Bert


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision('high')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--default_root_dir", type=str, default="outputs/")
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--add_stay_date", action="store_true")
    parser.add_argument("--add_rating", action="store_true")
    parser.add_argument("--add_pandemic", action="store_true")
    parser.add_argument("--multi_task", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--model", type=str, default="bert-cnn", choices=["bert-cnn", "bert", "bart-cnn", "bart"])
    parser.add_argument("--predict_with_label", action="store_true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    args = parser.parse_args()
    return args

def main():

    start = time.time()

    args = parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # Set random states
    seed_everything(args.seed)

    # Dataset
    dm = TripAdvisorDataModule(
        cache_dir=args.cache_dir,
        model_name_or_path=args.model_name_or_path,
        train_file = args.train_file,
        eval_file = args.eval_file,
        test_file = args.test_file,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        add_stay_date=args.add_stay_date,
        add_rating=args.add_rating,
        add_pandemic=args.add_pandemic,
        multi_task=args.multi_task,
        predict_with_label=args.predict_with_label,
    )
    dm.prepare_data()
    dm.setup()

    # Model
    if args.model == "bert-cnn":
        model = BertTextCNN(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels, 
            add_stay_date=args.add_stay_date,
            add_rating=args.add_rating,
            add_pandemic=args.add_pandemic,
            multi_task=args.multi_task,
        )
    elif args.model == "bert":
        model = Bert(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels, 
            add_stay_date=args.add_stay_date,
            add_rating=args.add_rating,
            add_pandemic=args.add_pandemic,
            multi_task=args.multi_task,
        )
    elif args.model == "bart-cnn":
        model = BartTextCNN(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels, 
            add_stay_date=args.add_stay_date,
            add_rating=args.add_rating,
            add_pandemic=args.add_pandemic,
            multi_task=args.multi_task,
        )
    elif args.model == "bart":
        model = Bart(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels, 
            add_stay_date=args.add_stay_date,
            add_rating=args.add_rating,
            add_pandemic=args.add_pandemic,
            multi_task=args.multi_task,
        )
    model_module = TripAdvisorModelModule(
        model, args.num_labels,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        multi_task=args.multi_task,
    )

    # Early stop
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience)

    # Set model save directory
    model_checkpoint = ModelCheckpoint(monitor="f1", dirpath=args.default_root_dir, mode="max", save_top_k=1)

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs, 
        default_root_dir=args.default_root_dir,
        accelerator=args.accelerator,
        deterministic=args.deterministic,
        devices=args.devices,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[early_stop_callback, model_checkpoint],
    )

    # Train
    if args.train_file:
        trainer.fit(model_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    # Predict
    if args.test_file:
        predictions = trainer.predict(dataloaders=dm.predict_dataloader(), ckpt_path="best")
        predictions = {k: torch.stack([o for batch in predictions for o in batch[k]]).cpu().tolist() for k in predictions[0].keys()}
        predictions["preds"] = [dm.id2label[pred] for pred in predictions["preds"]]

        if args.multi_task:
            for aspect in ["value", "location", "rooms", "service", "sleep", "cleanliness"]:
                predictions[f"{aspect}_preds"] = [pred+1 for pred in predictions[f"{aspect}_preds"]]

        label_names = ["labels", "value", "location", "rooms", "service", "sleep", "cleanliness"]
        predict_names = ["preds", "value_preds", "location_preds", "rooms_preds", "service_preds", "sleep_preds", "cleanliness_preds"]
        cr, cm = {}, {}
        for l_name, p_name in zip(label_names, predict_names):
            if l_name in dm.predict_dataset.features:
                if l_name == "labels":
                    labels = [dm.id2label[l.item()] for l in dm.predict_dataset[l_name].cpu()]
                else:
                    labels = [l.item()+1 for l in dm.predict_dataset[l_name].cpu()]
                preds = predictions[p_name]
                l_name = "type" if l_name == "labels" else l_name
                cr[l_name] = classification_report(labels, preds, output_dict=True, digits=4)
                cm[l_name] = confusion_matrix(labels, preds).tolist()

        predict_path = Path(args.output_dir) / "predictions.json"
        predict_path.write_text(json.dumps(predictions, indent=2))

        if len(cr) > 0:
            cr_path = Path(args.output_dir) / "classification_report.json"
            cr_path.write_text(json.dumps(cr, indent=2))

        if len(cm) > 0:
            cm_path = Path(args.output_dir) / "confusion_metric.json"
            cm_path.write_text(json.dumps(cm, indent=2))

    # Save arguments
    args_file = Path(args.output_dir) / "parameters.json"
    args_file.write_text(json.dumps(args.__dict__, indent=2))

    # Save processed time
    end = time.time()
    log_file = Path(args.output_dir) / "time.log"
    log_file.write_text(f"Process time: {end-start} seconds")


if __name__ == "__main__":
    main()