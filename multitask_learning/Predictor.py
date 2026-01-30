from DatasetModule import TripAdvisorDataModule
from ModelModule import TripAdvisorModelModule, BertTextCNN
from pytorch_lightning import Trainer
import torch
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run keyword experiment.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
    parser.add_argument("--output_dir", type=str, default="outputs/keywords_exp", help="Output directory for predictions")
    return parser.parse_args()


def main():

    args = parse_args()

    print(f"Running keyword experiment with test file: {args.test_file}")

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dm = TripAdvisorDataModule(
        model_name_or_path="roberta-base",
        test_file = args.test_file,
        max_seq_length=512,
        train_batch_size=16,
        eval_batch_size=16,
        add_stay_date=True,
        add_rating=True,
        add_pandemic=True,
        multi_task=True,
        predict_with_label=True,
    )
    dm.prepare_data()
    dm.setup()

    model = BertTextCNN(
        "roberta-base",
        num_labels=3, 
        add_stay_date=True,
        add_rating=True,
        add_pandemic=True,
        multi_task=True,
    )

    module = TripAdvisorModelModule.load_from_checkpoint(
        args.model_checkpoint,
        model=model,
        multi_task=True
    )

    # 設定 model 為 evaluation 模式
    module.eval()

    # 使用 Trainer 執行 predict
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    predictions = trainer.predict(module, dataloaders=dm.predict_dataloader())
    predictions = {k: torch.stack([o for batch in predictions for o in batch[k]]).cpu().tolist() for k in predictions[0].keys()}
    predictions["preds"] = [dm.id2label[pred] for pred in predictions["preds"]]

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


if __name__ == "__main__":
    main()