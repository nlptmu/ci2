import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CLASS_NAMES = ["family", "business", "friends"]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Model Explainer using LIME and SHAP.")
    parser.add_argument("--input_file", type=str, default="data/processed/test.json",
                        help="Path to the processed test data file.")
    parser.add_argument("--output_dir", type=str, default="data/processed/",
                        help="Directory to save the processed data.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model.")
    parser.add_argument("--num_instances", type=int, default=100,
                        help="Number of test instances to explain.")
    return parser.parse_args()


def load_data(file_path: str, num_instances: int):
    data = pd.read_json(file_path)
    return list(data["Review Content"])[:num_instances]


def explain_model_by_lime(model_path: str, test_data: list, output_dir: str):
    ## Load the tokenizer, model, and explainer 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)

    ## Define predictor function
    def predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        tensor_logits = outputs[0]
        probas = F.softmax(tensor_logits, dim=1).detach().numpy()
        return probas

    ## Explain each instance in the test data
    for i, text in enumerate(tqdm(test_data, desc="LIME")):
        Path(f"{output_dir}/test_{i}").mkdir(parents=True, exist_ok=True)
        explainer = LimeTextExplainer(class_names=CLASS_NAMES)
        exp = explainer.explain_instance(text, predictor, num_features=20, num_samples=500)
        exp.save_to_file(f"{output_dir}/test_{i}/lime.html")

    ## Delete the tokenizer, model, and explainer 
    del model
    del tokenizer
    del explainer


def explain_model_by_shap(model_path: str, test_data: list, output_dir: str):
    ## Load a transformers pipeline model
    model = transformers.pipeline("text-classification", model=model_path, top_k=None, device=DEVICE)

    ## Load the explainer
    explainer = shap.Explainer(model)

    ## Explain each instance in the test data
    for i, text in enumerate(tqdm(test_data, desc="SHAP")):
        Path(f"{output_dir}/test_{i}").mkdir(parents=True, exist_ok=True)
        shap_values = explainer([text])
        for class_name in CLASS_NAMES:
            shap_html = shap.plots.text(shap_values[0, :, class_name], display=False)
            Path(f"{output_dir}/test_{i}/shap_{class_name}.html").write_text(shap_html)
        Path(f"{output_dir}/test_{i}/shap.html").unlink()

    ## Delete the model and explainer
    del model
    del explainer


def main():
    args = parse_args()
    data = load_data(args.input_file, args.num_instances)
    explain_model_by_lime(args.model_path, data, args.output_dir)
    explain_model_by_shap(args.model_path, data, args.output_dir)


if __name__ == "__main__":
    main()