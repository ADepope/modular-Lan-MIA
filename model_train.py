import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
import torch
import datasets
import pandas as pd

class pretrained_models:

    def __init__(self, model_name, max_sequence_length, output_dir, log_dir, load_model_dir):
        # model_name = 'roberta_base'
        # max_sequence_length = 512
        # log_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/logfiles'
        # output_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/output'
            self.model_name = model_name
            self.max_sequence_length = max_sequence_length
            self.output_dir = output_dir
            self.log_dir = log_dir
            self.load_model_dir = load_model_dir
    
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
        
        # load model and tokenizer and define length of the text sequence, number of classes
        _, uniques = pd.factorize(train_data.to_pandas()["label"])
        print(f"num_classes = {len(uniques)}")
        model = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels = len(uniques))
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained(self.model_name, max_length = self.max_sequence_length)
        tokenize_data = lambda batch: tokenizer(batch['text'], padding=True, truncation=True)
        train_data = train_data.map(tokenize_data, batched=True, batch_size=len(train_data))
        test_data = test_data.map(tokenize_data, batched=True, batch_size=len(test_data))

        train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        ## if model is saved then load it

        if self.load_model_dir is not None:

            try:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(self.load_model_dir)
            except Exception as e:
                print(f"Error: {e}")
                print("Ensure you are using the output_dir where the model was saved.")

            print(model)

        else:
            
            ## else perform a fine-tuning of a model
            # define the training arguments
            training_args = transformers.TrainingArguments(
                output_dir = self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size = 4,
                gradient_accumulation_steps = 16,    
                per_device_eval_batch_size= 8,
                eval_strategy = "epoch",
                save_strategy = "epoch",
                load_best_model_at_end=True,
                warmup_steps=500,
                weight_decay=0.01,
                logging_steps = 8,
                fp16 = True,
                logging_dir=self.log_dir,
                dataloader_num_workers = 2,
                disable_tqdm=True # not diplaying a progress bar
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
            print("..train ended")
            # eval_res = trainer.evaluate()
            # print("..eval ended")

        return trainer, train_data, test_data
