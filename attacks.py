import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score, precision_score
import numpy as np
import torch
import datasets

class attack:

    def __init__(self, attack_name, num_trials):
        # attack_name = 'prediction_loss_based_mia'
        # num_trials = 500
            self.attack_name = attack_name
            self.num_trials = num_trials
            self.num_trials_per_cat = int( 0.5 * num_trials )
            self.mia_metrics = {}

    def calculate_mia_metrics(self, true_labels, predicted_labels):
        # Ensure the tensors are of type long (for classification)
        true_labels = true_labels.long()
        predicted_labels = predicted_labels.long()

        tp = torch.sum((true_labels == 1) & (predicted_labels == 1)).item()
        fp = torch.sum((true_labels == 0) & (predicted_labels == 1)).item()
        tn = torch.sum((true_labels == 0) & (predicted_labels == 0)).item()
        fn = torch.sum((true_labels == 1) & (predicted_labels == 0)).item()
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  
        tpr = recall  # recall is the same as TPR
        adv = tpr - fpr
        metrics = {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall (TPR)": recall,
            "F1 Score": f1,
            "FPR": fpr,
            "TPR": tpr,
            "Adv": adv
        }
        return metrics

    def print_mia_metrics(self):
        for metric, value in self.mia_metrics.items():
            print(f"{metric}: {value:.4f}")


    def perform_attack(self, trainer, train_data, test_data):

        shf_train_data = train_data.shuffle(seed=42).select(range(self.num_trials_per_cat))
        shf_test_data = test_data.shuffle(seed=42).select(range(self.num_trials_per_cat))
        attack_dataset = datasets.concatenate_datasets([shf_train_data, shf_test_data])

        if self.attack_name == 'prediction_loss_based_mia':

            # log_history = trainer.state.log_history
            # training_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
            # training_loss_mean = np.mean(training_loss)
            # print(trainer.state.log_history)
            last_record = trainer.state.log_history[-1]
            # print(last_record)
            training_loss = last_record['train_loss']

            predictions = trainer.predict(attack_dataset)

            # makes sense for classification tasks only
            logits = predictions.predictions
            labels = predictions.label_ids
            criterion = torch.nn.CrossEntropyLoss(reduction='none')  # using `reduction='none'` to get per-sample losses
            logits_tensor = torch.tensor(logits)
            labels_tensor = torch.tensor(labels)

            # calculate per-sample loss
            per_sample_loss = criterion(logits_tensor, labels_tensor)
            membership_pred = (per_sample_loss < training_loss).int()
            membership_true = torch.cat((torch.ones(self.num_trials_per_cat), torch.zeros(self.num_trials_per_cat)))
            self.mia_metrics = self.calculate_mia_metrics(membership_true, membership_pred)
            self.print_mia_metrics()

        elif self.attack_name == 'prediction_correctness_based_mia':

            predictions_output = trainer.predict(attack_dataset)
            predictions = predictions_output.predictions
            label_ids = predictions_output.label_ids
            predicted_labels = np.argmax(predictions, axis=1)
            membership_pred = torch.tensor( (predicted_labels == label_ids).astype(int) )
            membership_true = torch.cat((torch.ones(self.num_trials_per_cat), torch.zeros(self.num_trials_per_cat)))
            self.mia_metrics = self.calculate_mia_metrics(membership_true, membership_pred)
            self.print_mia_metrics()
        

        else:
    
            raise ValueError("Invalid attack was chosen!")


        return self.mia_metrics


