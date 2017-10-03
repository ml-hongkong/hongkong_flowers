from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, gc, time
import tensorflow as tf
from keras import backend as K

# custom modules
from core.keras.parser import ModelsDict, OptimsDict, LossesDict, MetricsDict, TransformsDict
from core.keras.callbacks import get_callbacks


class FlowerClassificationModel(object):
    def __init__(self, args):
        super(self.__class__, self).__init__()

        start_time = time.time()

        # params
        self.args = args
        self.train_dir = args.data_dir + 'train/'
        self.valid_dir = args.data_dir + 'valid/'
        self.test_dir = args.data_dir + 'test/'

        self.width = args.width
        self.height = args.height

        # suppress info and warning
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.verbose)

        # set which gpu to use
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "%i" % (3 - args.gpu)

        # this setting allocates memory dynamically
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        K.set_session(sess)

        # setup model
        model_choice, opt_choice, loss_choice, metric_choice = args.configs

        # remember these are classes
        Optim = OptimsDict[opt_choice]
        Loss = LossesDict[loss_choice]

        # for Keras, we use functional flowermodel!
        FlowerModel = ModelsDict[model_choice]

        # get folder names in train_dir, note that this is str!
        self.classes = [i for i in os.listdir(self.train_dir) if os.path.isdir(self.train_dir + i)]

        # count folders in data_dir
        self.num_classes = len(self.classes)

        # transform is an initialized ImageDataGenerator object for keras
        self.transform = TransformsDict[args.transform]

        # metric is a list of function!
        self.metric = MetricsDict[metric_choice]

        # initialize model
        self.model = FlowerModel(args, self.num_classes)

        # load previous model weights if toggled on
        if args.load_file:
            if os.path.isfile(args.load_file):
                self.model.load_weights(args.load_file, by_name=True)
                print("Successfully loaded weights from %s" % args.load_file)
            else:
                raise ValueError('Cannot find any model weights file in %s!' % args.load_file)

        # set up loss function, this is useless here but for the sake of completeness I leave it here.
        self.criterion = Loss

        # set up optimizer
        self.optimizer = Optim(lr=args.lr)

        # compile model
        self.model.compile(loss=self.criterion, optimizer=self.optimizer, metrics=self.metric)

        print ('Successfully loaded model. (%.4fs)' % (time.time() - start_time))
    def fit(self):

        # datasets and dataloader for training
        train_generator = self.transform.flow_from_directory(self.train_dir,
                                                             batch_size=self.args.batch_size,
                                                             target_size=(self.height, self.width),
                                                             classes=self.classes,
                                                             class_mode='categorical')

        valid_generator = self.transform.flow_from_directory(self.valid_dir,
                                                             batch_size=self.args.valid_batch_size,
                                                             target_size=(self.height, self.width),
                                                             classes=self.classes,
                                                             class_mode='categorical')

        train_step = train_generator.samples // self.args.batch_size
        valid_step = valid_generator.samples // self.args.valid_batch_size

        # create callbacks, remember to add data_gen to callbacks
        all_callbacks = get_callbacks(self.args)

        # fit the model
        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_step,
                                 epochs=self.args.num_epochs,
                                 callbacks=all_callbacks,
                                 validation_data=valid_generator,
                                 validation_steps=valid_step)

    def evaluate(self):

        # datasets and dataloader for test mode
        test_generator = self.transform.flow_from_directory(self.test_dir,
                                                            batch_size=self.args.test_batch_size,
                                                            target_size=(self.height, self.width),
                                                            classes=self.classes,
                                                            class_mode='categorical',
                                                            shuffle=False)

        test_step = test_generator.samples // self.args.test_batch_size

        loss = self.model.evaluate_generator(generator=test_generator,
                                             steps=test_step)

        print(loss)
        print(self.model.metrics_names)
        print('Test set loss = %.4f, accuracy = %.4f' % (loss[0], loss[1]))

        # save results if toggle on
        if self.args.save_test_result:
            import csv
            test_result_dict = {'loss' : loss[0], 'accuracy': loss[1]}
            with open(self.args.log_dir + 'test_result.csv', 'wb') as f:  # Just use 'w' mode in 3.x
                w = csv.DictWriter(f, test_result_dict.keys())
                w.writeheader()
                w.writerow(test_result_dict)

    def predict(self, x):

        return self.model.predict(x)

    def reset(self):
        print('Finished current experiment. Resetting model.')
        K.clear_session()
        gc.collect()
