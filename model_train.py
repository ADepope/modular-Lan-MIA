import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
import torch
import datasets

class pretrained_models:

    def __init__(self, model_name, max_sequence_length, output_dir, log_dir):
        # model_name = 'roberta_base'
        # max_sequence_length = 512
        # log_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/logfiles'
        # output_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/output'
            self.model_name = model_name
            self.max_sequence_length = max_sequence_length
            self.output_dir = output_dir
            self.log_dir = log_dir
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def fine_tune(self, train_data, test_data, num_epochs):
        
        # load model and tokenizer and define length of the text sequence
        model = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained(self.model_name, max_length = self.max_sequence_length)
        tokenize_data = lambda batch: tokenizer(batch['text'], padding=True, truncation=True)
        train_data = train_data.map(tokenize_data, batched=True, batch_size=len(train_data))
        test_data = test_data.map(tokenize_data, batched=True, batch_size=len(test_data))

        train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        # define the training arguments
        training_args = transformers.TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 16,    
            per_device_eval_batch_size= 8,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            disable_tqdm = False, 
            load_best_model_at_end=True,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps = 8,
            fp16 = True,
            logging_dir=self.log_dir,
            dataloader_num_workers = 2
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_data,
            eval_dataset=test_data
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"device = {device}")
        model.to(device)

        train_res = trainer.train()
        eval_res = trainer.evaluate()

        return trainer, train_data, test_data
