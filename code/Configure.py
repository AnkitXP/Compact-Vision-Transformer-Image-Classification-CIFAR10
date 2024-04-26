class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

model_configs = dotdict({
	"name": 'MyModel',
	"save_dir": '../saved_models/',
    "data_dir": '../data/',
    "result_dir": '../results/',
    "num_workers" : 8,     
    "img_size": 32,
    "in_channels": 3,
    "patch_size": 2,
    "depth": 7,
    "num_heads" : 4,
    "embed_dropout": 0.1,
    "transformer_dropout": 0.1,
    "dropout": 0.1,
    "num_classes": 10,
    "dim" : 64,
    "dim_head" : 64,
    "scale_dim" : 4
})

training_configs = dotdict({
    "num_epochs": 100,
	"learning_rate": 1e-3,
    "batch_size": 64,
    "save_interval": 10,
    "weight_decay": 1e-4
})
