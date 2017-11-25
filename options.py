from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

LIBRARIES = ['keras',    # 0
             'pytorch',  # 1
             ]

CONFIGS = [
#      model name,   optimizer,            loss              metric
[ "model_template",     "adam",        "CE_loss",    "simple_metric"],  # 0
[ "model_template",      "sgd",        "CE_loss",    "simple_metric"],  # 1
[ "resnet50_model",  "rmsprop",        "CE_loss",    "simple_metric"],  # 2
]
TRANSFORMS = [
    'to_tensor_only',      #0
    'to_resnet50_format',  #1
             ]


class Options(object):   # NOTE: shared across all modules
    def __init__(self, library=1, configs=0, transform=0):

        # TODO: NOT YET SET UP LOGGER
        self.verbose     = 2            # 0(warning) | 1(info) | 2(debug)

        # training signature for logging
        self.experiment  = "experiment"      # "experiment"
        self.number         = 0              # the current experiment number]
        self.timestamp      = str(datetime.datetime.today().strftime('%Y-%m-%d'))   # "yymmdd"

        # training configuration
        self.library          = library            # choose from LIBRARIES
        self.configs          = configs            # choose from CONFIGS
        self.transform        = transform          # choose from TRANSFORMS
        self.gpu              = None               # choose which GPU to use if any ( None for CPU)

        # general hyperparameters
        self.num_epochs       = 1000
        self.lr               = 1E-4
        self.batch_size       = 32
        self.valid_batch_size = 32
        self.test_batch_size  = 64
        self.grad_clip_norm   = 100000

        # regularizers hyperparameters
        self.dropout          = 0.5
        self.l2               = 0.01

        # set early stopping and logging toggles
        self.early_stopping   = True        # Toggle early_stopping
        self.patience         = 20           # number of epochs to consider before early stopping
        self.log_interval     = 1            # print every log_interval batch
        self.visualize        = False        # tensorboard for keras, visdom for pytorch

        # reduce lr on plateau
        self.reduce_lr        = 0.5         # Reduce lr on Plateau rate
        self.lr_patience      = 10           # patience for reducing lr
        self.min_lr           = 5E-6         # minimum lr that it can be reduced to

        # saving models
        self.save_best        = True         # saves the best model weights
        self.save_latest      = True         # saves the latest model weights
        self.log_history      = True         # logs down all the epoch results
        self.save_test_result = False        # saves test loss accuracy after evaluating

        # loading models when training
        self.load_file        = None        # load previous model weights when training if it exists

        # USE YOUR OWN DATA DIR: NEED ABSOLUTE PATH!
        self.data_dir         = './dataset/'
        self.log_dir          = "./logs/"
        self.ckpt_dir         = './checkpoints/'
        self.pretrained_file  = "./pretrained_models/fine-tuned-resnet50-weights.h5"

        # for the image size
        self.width     = 224
        self.height    = 224
        self.channel   = 3

        self.library   = LIBRARIES[self.library]
        self.configs   = CONFIGS[self.configs]
        self.transform = TRANSFORMS[self.transform]

        #----------------------------------------------------------------------------------------#
        # advance settings

        # pytorch settings
        if self.library == "pytorch":
            self.use_cuda           = True
            self.dataparallel       = False

        # keras settings
        if self.library == "keras":
            self.img_size  = (self.height, self.width, self.channel)

    def initializeModel(self, input_tensor=None):

        # choose model from library
        if self.library == 'pytorch':
            from core.pytorch.model import FlowerClassificationModel
        elif self.library == 'keras':
            from core.keras.model import FlowerClassificationModel
        else:
            raise ValueError('Library not supported.')

        return FlowerClassificationModel(self, input_tensor)
