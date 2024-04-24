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
    "num_encoders": 12,
    "embed_dropout": 0.,
    "transformer_dropout": 0.,
    "classifier_dropout": 0.3,
    "num_classes": 10,
    "hidden_channels" : 256,
    "head_channels" : 32
})

training_configs = dotdict({
    "num_epochs": 100,
	"learning_rate": 1e-3,
    "batch_size": 128,
    "save_interval": 20,
    "weight_decay": 1e-5
})
