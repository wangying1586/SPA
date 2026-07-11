from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from sklearn.metrics import confusion_matrix


def init_comet_experiment():
    """Initialize and return a new Comet experiment"""
    return Experiment(
        api_key="amErNSTjMM7BiByvvLlZnXy2n",
        project_name="pasa-extended",
        workspace="wangying1586"
    )


def log_hyperparameters(experiment, args):
    """Log hyperparameters to Comet experiment"""
    hyper_params = {
        "task_type": args.task_type,
        "feature_type": args.feature_type,
        "batch_size": args.batch_size,
        "warmup_epoch": args.warmup_epoch,
        "warmup_lr": args.warmup_lr,
        "epoch": args.epoch,
        "lr": args.lr,
        "early_stop": args.early_stop,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch_factor
    }
    experiment.log_parameters(hyper_params)


def get_task_labels(task_type):
    """Return labels based on task type"""
    task_labels = {
        11: ['Normal', 'Adventitious'],
        12: ['Normal', 'Rhonchi', 'Wheeze', 'Stridor', 'Coarse Crackle',
             'Fine Crackle', 'Wheeze+Crackle'],
        21: ['Normal', 'Poor Quality', 'Adventitious'],
        22: ['Normal', 'Poor Quality', 'CAS', 'DAS', 'CAS & DAS']
    }
    return task_labels.get(task_type, [])


def log_model_to_comet(experiment, model, model_name):
    """Log PyTorch model to Comet"""
    log_model(experiment, model=model, model_name=model_name)