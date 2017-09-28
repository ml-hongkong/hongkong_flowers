import os

# custom modules
from options import Options

def train():
    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # initialize model
    model = options.FlowerClassificationModel(options)

    # fit model
    model.fit()


if __name__=='__main__':
    train()
