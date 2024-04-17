model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
    "data_dir": '../data/',
    "model_dir":'../model_v1/'
}

training_configs = {
    "num_epochs": 40,
	"learning_rate": 0.01,
	"activation": 'gelu',
    "dropout_value": 1e-3,
    "num_heads": 8,
    "batch_size": 128,
    "save_interval": 10,
    "num_classes": 10,
    "patch_size": 4,
    "momentum": 0.9,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "random_seed": 42,
    "scheduler_milestones": (30, 35)
}
