from lightning.pytorch import LightningModule
import torchmetrics
from torch import nn
import torch

class ClassificationModel(LightningModule):
    def __init__(self,num_age_groups: int, num_classes: int = 2, learning_rate: float = 0.0001,
                sync_dist: bool = True, weight_decay: float = 0.0001,
                pos_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_age_groups = num_age_groups
        self.learning_rate = learning_rate
        self.sync_dist = sync_dist
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.model = None
        self.criterion = self.configure_loss()

        
        self._init_metrics()
        self._init_demographic_metrics()
        self.save_hyperparameters()

    def _init_demographic_metrics(self):
        self.gender_fairness = torchmetrics.classification.BinaryFairness(2)
        self.age_fairness = torchmetrics.classification.BinaryFairness(self.num_age_groups)

    def _init_metrics(self):
        """Initializes the metrics for the model.
        
        This method should be overridden in subclasses to define specific metrics.
        """
        if self.num_classes == 1:
            self.accuracy = torchmetrics.classification.BinaryAccuracy()
            self.precision = torchmetrics.classification.BinaryPrecision()
            self.recall = torchmetrics.classification.BinaryRecall()
            self.auroc = torchmetrics.classification.BinaryAUROC()
            self.f1 = torchmetrics.classification.BinaryF1Score()
            
        elif self.num_classes > 1:
            self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.precision = torchmetrics.classification.Precision(task="multiclass", num_classes=self.num_classes)
            self.recall = torchmetrics.classification.Recall(task="multiclass", num_classes=self.num_classes)
            self.auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=self.num_classes)
            self.f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=self.num_classes)

    def configure_loss(self):
        """Creates the loss function for the model.

        Returns:
            torch.nn.Module: The created loss function.
        """
        if self.num_classes == 1:
            return nn.BCEWithLogitsLoss()
        
        elif self.num_classes > 1:
            return nn.CrossEntropyLoss(
                label_smoothing=0.1
            )
        else:
            raise ValueError("num_classes should be greater than or equal to 1.")
    
    def forward(self, x):
        return self.model(x)

    def forward_features(self, x):
        return self.model.forward_features(x)
    
    def training_step(self, batch, batch_idx):
        """Defines the training step for the model.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        y_hat = self(batch.get('image'))
        loss = self.compute_loss(y_hat, batch.get('label'), 'train')

        self.compute_metrics(y_hat, batch.get('label'), 'train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Defines the validation step for the model.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        y_hat = self(batch.get('image'))
        loss = self.compute_loss(y_hat, batch.get('label'), 'val')

        self.compute_metrics(y_hat, batch.get('label'), 'val')
        self.compute_demographic_metrics(y_hat, batch, 'val')
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Defines the test step for the model.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        y_hat = self(batch.get('image'))
        loss = self.compute_loss(y_hat, batch.get('label'), 'test')

        self.compute_metrics(y_hat, batch.get('label'), 'test')
        self.compute_demographic_metrics(y_hat, batch, 'test')
        
        return loss

    def compute_loss(self, y_hat, y, mode):
        """Computes the loss for the model.

        Args:
            y_hat (torch.Tensor): The predicted values.
            y (torch.Tensor): The true values.
            mode (str): The mode of operation ('train', 'val', 'test').

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = self.criterion(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    def compute_metrics(self, y_hat, y, mode):
        """Computes the metrics for the model.
        Args:
            y_hat (torch.Tensor): The predicted values.
            y (torch.Tensor): The true values.
            mode (str): The mode of operation ('train', 'val', 'test').
        """
        acc = self.accuracy(y_hat, y.int())
        prec = self.precision(y_hat, y.int())
        rec = self.recall(y_hat, y.int())
        auroc = self.auroc(y_hat, y.int())
        f1 = self.f1(y_hat, y.int())

        self.log(f'{mode}_acc', acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_prec', prec, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_recall', rec, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_auroc', auroc, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_f1', f1, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
    
    def _compute_metrics_for_group(self, y_hat, y, group, group_name, mode):
        # Store AUROC values for each group
        auroc_values = {}

        for group_value in torch.unique(group):
            mask = group == group_value
            if mask.any():
                auroc = self.auroc(y_hat[mask], y[mask].int())
                f1 = self.f1(y_hat[mask], y[mask].int())
                acc = self.accuracy(y_hat[mask], y[mask].int())
                self.log(f'{mode}_{group_name}_auroc_{group_value.item()}', auroc, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
                self.log(f'{mode}_{group_name}_f1_{group_value.item()}', f1, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
                self.log(f'{mode}_{group_name}_acc_{group_value.item()}', acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
                # Store the AUROC value for this group
                auroc_values[group_value.item()] = auroc
            else:
                self.log(f'{mode}_{group_name}_auroc_{group_value.item()}', 0, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
                self.log(f'{mode}_{group_name}_f1_{group_value.item()}', 0, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
                self.log(f'{mode}_{group_name}_acc_{group_value.item()}', 0, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        # Calculate min AUROC / max AUROC ratio
        if auroc_values:
            min_auroc = min(auroc_values.values())
            max_auroc = max(auroc_values.values())
            ratio = min_auroc / max_auroc if max_auroc != 0 else 0
        else:
            ratio = 0

        # Log the min/max AUROC ratio
        self.log(f'{mode}_{group_name}_min_max_auroc_ratio', ratio, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

    def compute_demographic_metrics(self, y_hat, batch, mode):
        """Computes the demographic metrics for the model.

        Args:
            y (torch.Tensor): The true values.
            mode (str): The mode of operation ('train', 'val', 'test').
        """
        self._compute_metrics_for_group(y_hat, batch.get('label'), batch.get('gender'), 'gender', mode)
        self._compute_metrics_for_group(y_hat, batch.get('label'), batch.get('age'), 'age', mode)

        # # Convert one-hot encoded predictions to class format if necessary
        # if y_hat.shape[1] == self.num_classes:
        #     y_hat = y_hat.argmax(dim=1)

        # gender_fairness = self.gender_fairness(y_hat, batch.get('label').int(), batch.get('gender'))
        # age_fairness = self.age_fairness(y_hat, batch.get('label').int(), batch.get('age'))

        # for key, value in gender_fairness.items():
        #     self.log(f'test_gender-fairness_{key}', value.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        # for key, value in age_fairness.items():
        #     self.log(f'test_age-fairness_{key}', value.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
