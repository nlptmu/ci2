from typing import List
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset


class TripAdvisorDataModule:

    def __init__(
        self,
        cache_dir: str,
        model_name_or_path: str,
        train_file: str = None,
        eval_file: str = None,
        test_file: str = None,
        max_seq_length: int = 512,
        train_batch_size: int = 16,
        eval_batch_size: int = 16,
        add_stay_date: bool = False,
        add_rating: bool = False,
        add_pandemic: bool = False,
        multi_task: bool = False,
        predict_with_label: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.add_stay_date = add_stay_date
        self.add_rating = add_rating
        self.add_pandemic = add_pandemic
        self.multi_task = multi_task
        self.predict_with_label = predict_with_label

    def prepare_data(self):
        if self.train_file:
            self.train_data = pd.read_json(Path(self.train_file))

        if self.eval_file:
            self.eval_data = pd.read_json(Path(self.eval_file))

        if self.test_file:
            self.test_data = pd.read_json(Path(self.test_file))

    def setup(self):
        self.year_cat = list(range(2017, 2024)) # year is between 2017 and 2023
        self.month_cat = list(range(1, 13))     # month is between 1 and 12
        self.rating_cat = list(range(1, 6))     # rating ranges from 1 to 5

        self.label_cat = ["family", "business", "friends"]
        self.label2id = {l: i for i, l in enumerate(self.label_cat)}
        self.id2label = {i: l for i, l in enumerate(self.label_cat)}

        if self.multi_task:
            self.tasks = ["Value", "Location", "Rooms", "Service", "Sleep Quality", "Cleanliness"]
        
        if self.add_pandemic:
            self.pandemic_cat = ["Pre-Pandemic", "During-Pandemic", "Post-Pandemic"]

        if self.train_file:
            self.train_dataset = self.generate_dataset(self.train_data)
        
        if self.eval_file:
            self.eval_dataset = self.generate_dataset(self.eval_data)

        if self.test_file:
            self.predict_dataset = self.generate_dataset(self.test_data, with_label=self.predict_with_label)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.eval_batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.eval_batch_size, num_workers=4)

    def generate_dataset(self, data: pd.DataFrame, with_label: bool = True) -> Dataset:
        data_columns = ["Review Title", "Review Content"]

        if with_label:
            data_columns.extend(["Travel Type"])

        if self.add_stay_date:
            data_columns.extend(["Stay Year", "Stay Month"])

        if self.add_rating:
            data_columns.extend(["Overall Rating"])

        if self.multi_task and with_label:
            data_columns.extend(["Value", "Location", "Rooms", "Service", "Sleep Quality", "Cleanliness"])

        if self.add_pandemic:
            data_columns.extend(["Pandemic"])

        dataset = Dataset.from_dict({col: [row[col] for i, row in data.iterrows()] for col in data_columns})
        dataset = dataset.map(self.convert_to_features, batched=True, remove_columns=data_columns)
        dataset.set_format(type="torch", columns=dataset.column_names)

        return dataset

    def convert_to_features(self, examples):
        texts = [f"{title}. {content}" for title, content in zip(examples["Review Title"], examples["Review Content"])]
        features = self.tokenizer(
            texts, max_length=self.max_seq_length, padding="max_length", truncation=True)
        
        if "Travel Type" in examples:
            features["labels"] = [self.label2id[self.traveler_type(label)] for label in examples["Travel Type"]]

        if self.add_stay_date:
            features["stay_year"] = [self.onehot_encoding(val, self.year_cat) for val in examples["Stay Year"]]
            features["stay_month"] = [self.onehot_encoding(val, self.month_cat) for val in examples["Stay Month"]]
        
        if self.add_rating:
            features["rating"] = [
                self.onehot_encoding(val, self.rating_cat) for val in examples["Overall Rating"]]

        if self.multi_task and "Value" in examples:
            for task in self.tasks:
                task_name = "sleep" if task == "Sleep Quality" else task.lower()
                features[task_name] = [label-1 for label in examples[task]] # normalize to 0-4
        
        if self.add_pandemic:
            features["pandemic"] = [
                self.onehot_encoding(val, self.pandemic_cat) for val in examples["Pandemic"]]

        return features

    def onehot_encoding(self, x: int, categories: List) -> List[int]:
        rtn = [0 for _ in range(len(categories))]
        try:
            rtn[categories.index(x)] = 1
        except:
            rtn[-1] = 1 # if not found, set to the last category
        return rtn
    
    def traveler_type(self, text: str) -> str:
        if text == "Traveled with family":
            return "family"
        elif text == "Traveled on business":
            return "business"
        elif text == "Traveled as a couple":
            return "friends"
        elif text == "Traveled with friends":
            return "friends"
        elif text == "Traveled solo":
            return "business"