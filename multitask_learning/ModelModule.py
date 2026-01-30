from typing import Optional, List
from dataclasses import dataclass

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn import Module, Dropout, Linear, CrossEntropyLoss
from torchmetrics.functional import f1_score
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup


@dataclass
class ModelOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    aspect_loss: Optional[torch.FloatTensor] = None
    value_logits: torch.FloatTensor = None
    location_logits: torch.FloatTensor = None
    rooms_logits: torch.FloatTensor = None
    service_logits: torch.FloatTensor = None
    sleep_logits: torch.FloatTensor = None
    cleanliness_logits: torch.FloatTensor = None
    

class TripAdvisorModelModule(LightningModule):
    def __init__(
        self,
        model,
        num_labels,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        multi_task: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('train_loss', outputs.loss)
        if outputs.aspect_loss is not None:
            self.log('train_aspect_loss', outputs.aspect_loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val_loss", outputs.loss)
        if outputs.aspect_loss is not None:
            self.log('val_aspect_loss', outputs.aspect_loss)
        self.log("f1", f1_score(outputs.logits.argmax(-1), batch["labels"], task="multiclass", num_classes=self.hparams.num_labels))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        rtn =  {"preds": torch.argmax(outputs.logits, axis=-1)}
        if self.hparams.multi_task:
            rtn.update({
                "value_preds": torch.argmax(outputs.value_logits, axis=-1),
                "location_preds": torch.argmax(outputs.location_logits, axis=-1),
                "rooms_preds": torch.argmax(outputs.rooms_logits, axis=-1),
                "service_preds": torch.argmax(outputs.service_logits, axis=-1),
                "sleep_preds": torch.argmax(outputs.sleep_logits, axis=-1),
                "cleanliness_preds": torch.argmax(outputs.cleanliness_logits, axis=-1),
            })
        return rtn
            
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]


class BertTextCNN(Module):

    def __init__(
        self, 
        model_name_or_path: str,
        cache_dir: str,
        num_labels: int,
        add_stay_date: bool = False,
        add_rating: bool = False,
        add_pandemic: bool = False,
        multi_task: bool = False,
        label_weight: float = 0.5,
        aspect_weight: float = 0.2,
        num_filters: int = 128,
        filter_sizes: List = [3, 4, 5],
        num_year: int = 7,
        num_month: int = 12,
        num_rating: int = 5,
        num_pandemic: int = 3,
        second_hidden_size: int = 128,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.add_stay_date = add_stay_date
        self.add_rating = add_rating
        self.add_pandemic = add_pandemic
        self.multi_task = multi_task
        self.label_weight = label_weight
        self.aspect_weight = aspect_weight
        self.num_rating = num_rating
   
        # Bert
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)

        self.encoder.embeddings.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

        # TextCNN
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(1, num_filters, (K, config.hidden_size)) for K in filter_sizes])
        self.classifier = torch.nn.Linear(len(filter_sizes) * num_filters, config.num_labels)

        # Dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = Dropout(classifier_dropout)

        # Feature size
        feature_size = len(filter_sizes) * num_filters
        if add_stay_date:
            feature_size += (num_year + num_month)
        if add_rating:
            feature_size += num_rating
        if add_pandemic:
            feature_size += num_pandemic

        self.fc = Linear(feature_size, second_hidden_size)
        self.classifier = Linear(second_hidden_size, num_labels)

        if multi_task:
            self.value_classifier = Linear(second_hidden_size, num_rating)
            self.location_classifier = Linear(second_hidden_size, num_rating)
            self.rooms_classifier = Linear(second_hidden_size, num_rating)
            self.service_classifier = Linear(second_hidden_size, num_rating)
            self.sleep_classifier = Linear(second_hidden_size, num_rating)
            self.cleanliness_classifier = Linear(second_hidden_size, num_rating)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stay_year: Optional[torch.Tensor] = None,
        stay_month: Optional[torch.Tensor] = None,
        pandemic: Optional[torch.Tensor] = None,
        rating: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None,
        rooms: Optional[torch.Tensor] = None,
        service: Optional[torch.Tensor] = None,
        sleep: Optional[torch.Tensor] = None,
        cleanliness: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ModelOutput:

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[0]

        pooled_output = pooled_output.unsqueeze(1)
        pooled_output = [F.relu(conv(pooled_output)).squeeze(3) for conv in self.conv]
        pooled_output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in pooled_output]
        pooled_output = torch.cat(pooled_output, 1)
        pooled_output = self.dropout(pooled_output)

        if self.add_stay_date:
            pooled_output = torch.cat([pooled_output, stay_year, stay_month], axis=-1)
        if self.add_rating:
            pooled_output = torch.cat([pooled_output, rating], axis=-1)
        if self.add_pandemic:
            pooled_output = torch.cat([pooled_output, pandemic], axis=-1)

        pooled_output = self.fc(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss()
        loss = None
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        output = ModelOutput(loss=loss, logits=logits)

        if self.multi_task:
            value_logits = self.value_classifier(pooled_output)
            location_logits = self.location_classifier(pooled_output)
            rooms_logits = self.rooms_classifier(pooled_output)
            service_logits = self.service_classifier(pooled_output)
            sleep_logits = self.sleep_classifier(pooled_output)
            cleanliness_logits = self.cleanliness_classifier(pooled_output)

            output.value_logits = value_logits
            output.location_logits = location_logits
            output.rooms_logits = rooms_logits
            output.service_logits = service_logits
            output.sleep_logits = sleep_logits
            output.cleanliness_logits = cleanliness_logits

            if value is not None:
                aspect_loss = (
                    loss_fct(value_logits.view(-1, self.num_rating), value.view(-1)) +
                    loss_fct(location_logits.view(-1, self.num_rating), location.view(-1)) +
                    loss_fct(rooms_logits.view(-1, self.num_rating), rooms.view(-1)) +
                    loss_fct(service_logits.view(-1, self.num_rating), service.view(-1)) +
                    loss_fct(sleep_logits.view(-1, self.num_rating), sleep.view(-1)) +
                    loss_fct(cleanliness_logits.view(-1, self.num_rating), cleanliness.view(-1)))
                total_loss = self.label_weight * loss + self.aspect_weight * aspect_loss
                output.loss = total_loss
                output.aspect_loss = aspect_loss

        return output
    

class Bert(Module):

    def __init__(
        self, 
        model_name_or_path: str,
        cache_dir: str,
        num_labels: int,
        add_stay_date: bool = False,
        add_rating: bool = False,
        add_pandemic: bool = False,
        multi_task: bool = False,
        label_weight: float = 0.5,
        aspect_weight: float = 0.2,
        num_year: int = 7,
        num_month: int = 12,
        num_rating: int = 5,
        num_pandemic: int = 3,
        second_hidden_size: int = 128,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.add_stay_date = add_stay_date
        self.add_rating = add_rating
        self.add_pandemic = add_pandemic
        self.multi_task = multi_task
        self.label_weight = label_weight
        self.aspect_weight = aspect_weight
        self.num_rating = num_rating
   
        # Bert
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)

        # Dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = Dropout(classifier_dropout)

        # Feature size
        feature_size = config.hidden_size
        if add_stay_date:
            feature_size += (num_year + num_month)
        if add_rating:
            feature_size += num_rating
        if add_pandemic:
            feature_size += num_pandemic

        self.fc = Linear(feature_size, second_hidden_size)
        self.classifier = Linear(second_hidden_size, num_labels)

        if multi_task:
            self.value_classifier = Linear(second_hidden_size, num_rating)
            self.location_classifier = Linear(second_hidden_size, num_rating)
            self.rooms_classifier = Linear(second_hidden_size, num_rating)
            self.service_classifier = Linear(second_hidden_size, num_rating)
            self.sleep_classifier = Linear(second_hidden_size, num_rating)
            self.cleanliness_classifier = Linear(second_hidden_size, num_rating)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stay_year: Optional[torch.Tensor] = None,
        stay_month: Optional[torch.Tensor] = None,
        pandemic: Optional[torch.Tensor] = None,
        rating: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None,
        rooms: Optional[torch.Tensor] = None,
        service: Optional[torch.Tensor] = None,
        sleep: Optional[torch.Tensor] = None,
        cleanliness: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ModelOutput:

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.add_stay_date:
            pooled_output = torch.cat([pooled_output, stay_year, stay_month], axis=-1)
        if self.add_rating:
            pooled_output = torch.cat([pooled_output, rating], axis=-1)
        if self.add_pandemic:
            pooled_output = torch.cat([pooled_output, pandemic], axis=-1)

        pooled_output = self.fc(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss()
        loss = None
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        output = ModelOutput(loss=loss, logits=logits)

        if self.multi_task:
            value_logits = self.value_classifier(pooled_output)
            location_logits = self.location_classifier(pooled_output)
            rooms_logits = self.rooms_classifier(pooled_output)
            service_logits = self.service_classifier(pooled_output)
            sleep_logits = self.sleep_classifier(pooled_output)
            cleanliness_logits = self.cleanliness_classifier(pooled_output)

            output.value_logits = value_logits
            output.location_logits = location_logits
            output.rooms_logits = rooms_logits
            output.service_logits = service_logits
            output.sleep_logits = sleep_logits
            output.cleanliness_logits = cleanliness_logits

            if value is not None:
                aspect_loss = (
                    loss_fct(value_logits.view(-1, self.num_rating), value.view(-1)) +
                    loss_fct(location_logits.view(-1, self.num_rating), location.view(-1)) +
                    loss_fct(rooms_logits.view(-1, self.num_rating), rooms.view(-1)) +
                    loss_fct(service_logits.view(-1, self.num_rating), service.view(-1)) +
                    loss_fct(sleep_logits.view(-1, self.num_rating), sleep.view(-1)) +
                    loss_fct(cleanliness_logits.view(-1, self.num_rating), cleanliness.view(-1)))
                total_loss = self.label_weight * loss + self.aspect_weight * aspect_loss
                output.loss = total_loss
                output.aspect_loss = aspect_loss

        return output