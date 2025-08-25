"""Main script for training and evaluating Vision Transformer models.

This script serves as the main entry point for configuring and running
training or testing pipelines for various medical imaging datasets using
PyTorch Lightning and a Vision Transformer architecture. It handles command-line
argument parsing, dataset and model instantiation, trainer configuration,
optional learning rate tuning, and execution of training and/or testing loops.
"""
import argparse
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner.tuning import Tuner

from src.datamodule.dataset.vision import OdirModule, BrsetModule
from src.datamodule.dataset.chest import CheXpertModule
from src.datamodule.dataset.skin import Ham10000Module
from src.model.classification import VisionTransformerModel
from src.config import config

# Map dataset names to their corresponding DataModule classes
DATASET_MODULES = {
    "brset": BrsetModule,
    "odir": OdirModule,
    "chexpert": CheXpertModule,
    "ham10000": Ham10000Module,
}

def main(args: argparse.Namespace):
    """Sets up and runs the training or testing pipeline.

    Based on the provided command-line arguments, this function:
    1. Validates the chosen dataset.
    2. Loads dataset-specific configurations.
    3. Instantiates the appropriate DataModule and Model (VisionTransformerModel).
       - Retrieves the model's required transform and passes it to the DataModule.
    4. Configures PyTorch Lightning Callbacks (ModelCheckpoint) and the Trainer.
    5. Optionally performs learning rate tuning using Lightning's Tuner.
    6. Runs either the training loop (followed by testing) or just the testing loop.

    Args:
        args: An argparse.Namespace object containing the parsed command-line
              arguments, including dataset choice, hyperparameters (batch size,
              epochs, learning rate, weight decay), model settings (pretrain),
              data handling options (fraction, num_groups, augment_train),
              execution flags (tune, test), and logging details (name).

    Raises:
        ValueError: If the specified dataset name in `args.dataset` is not
                    found in the `DATASET_MODULES` mapping.
    """
    # Validate dataset choice
    if args.dataset not in DATASET_MODULES:
        raise ValueError(f"Invalid dataset '{args.dataset}'. Choose from {list(DATASET_MODULES.keys())}")

    # Get dataset-specific configuration
    dataset_config = config['data'][args.dataset]
    num_classes = dataset_config['num_classes']

    # Instantiate model first to get its transform
    model_id = config['model'][args.model_name]['id']

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_weights_only=True # Save only weights to save space
    )

    # Configure the PyTorch Lightning Trainer
    print("Configuring Trainer...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=TensorBoardLogger('output/logs/', name=args.name),
        callbacks=[checkpoint_callback],
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Configure and initialize the actual model instance to be used for training/testing
    # Initialize the actual model instance within the Trainer's context.
    # This is important for compatibility with certain Lightning strategies (e.g., DDP).
    print("Initializing model within Trainer context...")

    

    with trainer.init_module():
         model = VisionTransformerModel(
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_age_groups=args.num_groups,
            model_id=model_id,
            pretrained=args.pretrained,
            max_epochs=args.max_epochs,
            weights_path=config['model'][args.model_name]['weights_path'],
            freeze_layers=args.freeze_layers,
            loss=args.loss,
            fairness_weight=args.fairness_weight
        )
        

    # Instantiate the correct DataModule based on the dataset argument
    print(f"Instantiating DataModule for dataset: {args.dataset}")
    datamodule_class = DATASET_MODULES[args.dataset]
    datamodule = datamodule_class(
        data_dir=dataset_config['data_path'],
        image_data_dir=dataset_config['image_data_path'],
        batch_size=args.batch_size,
        model_transform=model.transform, # Pass the transform from the model
        augment_train=args.augment_train, # Pass the augmentation flag
        fraction=args.fraction,
        num_workers=dataset_config['num_workers'],
        task=dataset_config['task'],
        num_groups=args.num_groups,
        random_seed=args.random_seed,
    )

    if args.test:
        print("Running in test-only mode...")
        # Assumes a checkpoint exists from a previous run or is specified.
        # Using ckpt_path='best' relies on ModelCheckpoint saving the best model during a 'fit' call.
        # For true standalone testing, a specific --checkpoint_path argument should be used.
        trainer.test(
            model=model,
            datamodule=datamodule,
            # ckpt_path=args.checkpoint_path or "best" # Example if adding checkpoint arg
            ckpt_path="best" # Loads the best checkpoint saved during fit
        )
    else:
        if args.tune:
            print("Starting learning rate tuning...")
            tuner = Tuner(trainer)
            # tuner.scale_batch_size(model, datamodule=datamodule, mode="power", init_val=4)
            lr_finder_results = tuner.lr_find(model, datamodule=datamodule)
            suggested_lr = model.learning_rate
            print(f"Suggested learning rate: {suggested_lr}; Suggestion from results: {lr_finder_results.suggestion()}")

        print("Starting training...")
        trainer.fit(
            model,
            datamodule=datamodule,
            # ckpt_path=args.resume_from_checkpoint # Uncomment and add arg if resuming needed
        )

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Train or test a Vision Transformer model on various medical imaging datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help message
    )

    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'], help='Input batch size')
    parser.add_argument('--max_epochs', type=int, default=config['training']['epochs'], help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=config['training']['learning_rate'], help='Initial learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=config['training']['weight_decay'], help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--augment_train', action='store_true', default=config['training'].get('augment_train', False), help='Apply data augmentation during training.')
    parser.add_argument('--tune', action='store_true', default=config['training']['tune'], help='Enable learning rate tuning before training.') # Changed to action='store_true'
    parser.add_argument('--num_nodes', type=int, default=config['training']['num_nodes'], help='Number of nodes for distributed training.')
    parser.add_argument('--strategy', type=str, default=config['training']['strategy'], help='Distributed training strategy (e.g., ddp, dp).')
    parser.add_argument('--accumulate_grad_batches', type=int, default=config['training']['accumulate_grad_batches'], help='Number of batches to accumulate gradients before updating the model.')
    parser.add_argument('--fraction', type=float, default=config['training']['fraction'], help='Fraction of the dataset to use for training. 1.0 means full dataset.')
    parser.add_argument('--freeze_layers', type=int, default=config['training'].get('freeze_layers', 0), help='Number of layers to freeze (0 = no freezing, 1 = freeze all but head, 2 = freeze all but head and one more layer, etc.)')
    parser.add_argument('--loss', type=str, default=config['training']['loss'], choices=['ce', 'fair'], help='Loss function to use for training.')
    parser.add_argument('--fairness_weight', type=float, default=config['training']['fairness_weight'], help='Weight for fairness loss.')
    # --- Data Configuration ---
    parser.add_argument('--dataset', type=str, default=config['data']['dataset'], choices=list(DATASET_MODULES.keys()), help='Select the dataset to use.')
    parser.add_argument('--num_groups', type=int, default=config['data']['num_groups'], help='Number of demographic groups (e.g., age) for fairness analysis.')
    parser.add_argument('--random_seed', type=int, default=config['data']['random_seed'], help='Random seed for reproducibility.')
    
    # --- Model Configuration ---
    parser.add_argument('--model_name', type=str, default=config['model']['name'], help='Identifier for the specific model architecture (used with config.yml).')
    parser.add_argument('--pretrained', action='store_true', default=config['model']['pretrained'], help='Use a pretrained model. If False, train from scratch.')

    # --- Logging and Execution Control ---
    parser.add_argument('--name', type=str, default=config['log']['name'], help='Experiment name used for TensorBoard logging directory.')
    parser.add_argument('--test', action='store_true', help='Run in test-only mode. Requires a trained checkpoint (uses "best" by default).')

    args = parser.parse_args()
    main(args)
