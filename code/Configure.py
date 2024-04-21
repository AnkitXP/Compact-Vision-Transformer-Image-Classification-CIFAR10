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
    "num_workers" : 8,     
    "img_size": 32,
    "in_channels": 3,
    "patch_size": 2,
    "num_encoders": 8,
    "num_heads": 12,
    "embed_dropout": 0.,
    "transformer_dropout": 0.,
    "classifier_dropout": 0.,
    "num_classes": 10,
    "hidden_channels" : 256,
    "head_channels" : 32
})

training_configs = dotdict({
    "num_epochs": 10,
	"learning_rate": 0.001,
    "batch_size": 128,
    "save_interval": 4,    
    "momentum": 0.9,
    "weight_decay": 0.,
    "betas": (0.9, 0.99)
})
