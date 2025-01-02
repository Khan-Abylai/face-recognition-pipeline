from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = True
config.output = "/face_data/out_combined"

config.embedding_size = 512

config.sample_rate = 0.2
config.interclass_filtering_threshold = 0

config.fp16 = True
config.batch_size = 128


config.verbose = 10000
config.frequent = 10


config.rec = "/face_data/combined_dataset"
config.num_classes = 770300
config.num_image = 14393183
config.num_epoch = 25
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.num_workers = 8

# For Large Scale Dataset, such as WebFace42M
config.dali = False
config.dali_aug = False

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# For SGD
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 0.0005

config.save_all_states = True