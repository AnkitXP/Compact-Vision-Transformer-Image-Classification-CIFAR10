class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

model_configs = dotdict({
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
    "data_dir": '../data/',
    "model_dir": '../model_v1/',
    "result_dir": '../results/',
    
    
    "img_size": 32,
    "in_channels": 3,
    "patch_size": 4,
    "num_encoders": 12,
    "num_heads": 12,
    "dropout_value": 0.1,
    "num_classes": 10,
    "mlp_size": 3072
})

training_configs = dotdict({
    "num_epochs": 40,
	"learning_rate": 0.001,
	"activation": 'gelu',
    "batch_size": 128,
    "save_interval": 10,    
    "momentum": 0.9,
    "weight_decay": 0.,
    "betas": (0.9, 0.999),
    "random_seed": 42,
    "scheduler_milestones": (30, 35)
})
