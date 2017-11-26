from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, sys, os
import torch
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from core.pytorch.parser import ModelsDict, OptimsDict, LossesDict, MetricsDict, TransformsDict
from core.pytorch.utils import AverageMeter

class FlowerClassificationModel(object):
    def __init__(self, args, input_tensor=None):
        super(self.__class__, self).__init__()

        # params
        self.train_dir = args.data_dir + 'train/'
        self.valid_dir = args.data_dir + 'valid/'
        self.test_dir = args.data_dir + 'test/'
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.grad_clip_norm = args.grad_clip_norm
        self.batch_size = args.batch_size
        self.valid_batch_size = args.valid_batch_size
        self.test_batch_size = args.test_batch_size

        # this is for removing torch dependency in options.py
        if args.use_cuda:
            self.cuda = torch.cuda.is_available()
        else:
            self.cuda = False

        # set up visdom
        if args.visualize:
            self.vis = visdom.Visdom()

        # setup model
        model_choice, opt_choice, loss_choice, metric_choice = args.configs

        # remember these are classes
        FlowerModel = ModelsDict[model_choice]
        Optim = OptimsDict[opt_choice]
        Loss = LossesDict[loss_choice]

        # but metric is a function!
        self.metric = MetricsDict[metric_choice]

        # get folder names in train_dir, note that this is in str!
        self.classes = [i for i in os.listdir(self.train_dir) if os.path.isdir(self.train_dir + i)]

        # count folders in data_dir
        self.num_classes = len(self.classes)

        # parsed transform function for dataset
        self.transform = TransformsDict[args.transform]

        # initialize model
        self.model = FlowerModel(args, self.num_classes)

        # set up data parallel and cuda for model
        if self.cuda:
            if args.dataparallel:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()

        self.criterion = Loss()
        self.optimizer = Optim(self.model.parameters(), lr=args.lr)

    def train(self, train_dataloader, epoch):

        # switch to train mode
        self.model.train()

        end = time.time()  # this one is for each update, will be updated every print interval.
        start_time = time.time()  # this one is for each epoch.

        # logs for loss, grad_norm and time
        losses = AverageMeter()
        grad_norm = AverageMeter()
        load_time = AverageMeter()
        total_time = AverageMeter()

        for i_batch, batch in enumerate(train_dataloader):

            load_time.update(time.time() - end) # logs the time taken for loading data

            inputs = Variable(batch[0], requires_grad=False)
            labels = Variable(batch[1], requires_grad=False)
            labels = torch.squeeze(labels)

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward pass
            outputs = self.model(inputs) # batch size * num_class

            # calculate softmax loss
            loss = self.criterion(outputs, labels)
            loss_sum = loss.data[0]
            losses.update(loss_sum)

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            grad_norm.update(torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip_norm))

            # update weights
            self.optimizer.step()

            if self.cuda:
                torch.cuda.synchronize()

            if i_batch % self.log_interval == self.log_interval-1:
                sys.stdout.write("\r[%d, %5d] loss: %.3f, clipped grad norm: %.3f, load: %.2fs, total: %.2fs" %
                      (epoch+1, i_batch+1, losses.avg, grad_norm.avg, load_time.avg * self.log_interval, total_time.avg * self.log_interval))
                sys.stdout.flush()
                grad_norm.reset() # reset here so only print the avg in print interval
                load_time.reset()
                total_time.reset()

            total_time.update(time.time() - end)

            # end time
            end = time.time()

        print("\n\nEpoch %i: Train Loss = %.4f, Total Time = %.4fs" % (epoch, losses.avg, time.time() - start_time))

    def validate(self, valid_dataloader, epoch):

        # switch to eval mode
        self.model.eval()

        start_time = time.time()

        losses = AverageMeter()
        total_acc = AverageMeter()

        for i_batch, batch in enumerate(valid_dataloader):

            inputs = Variable(batch[0], requires_grad=False)
            labels = Variable(batch[1], requires_grad=False)
            labels = torch.squeeze(labels)

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = self.model(inputs)

            # calculate accuracy
            accuracy = self.metric(outputs, labels)

            # calculate softmax loss
            loss = self.criterion(outputs, labels)
            loss = loss.data[0]

            losses.update(loss)
            total_acc.update(accuracy)

            if self.cuda:
                torch.cuda.synchronize()

        print("Epoch %i: Valid Accuracy = %.4f, Valid Loss = %.4f, Total Time = %.4fs \n" % (epoch,total_acc.avg, losses.avg, time.time() - start_time))

    def fit(self):

        # datasets and dataloader for train mode
        train_dataset = ImageFolder(self.train_dir, transform=self.transform)
        valid_dataset = ImageFolder(self.valid_dir, transform=self.transform)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=self.valid_batch_size,
                                      shuffle=False)

        for epoch in range(self.num_epochs):
            print("---------------------------------------------------------")
            print('Epoch %d:' % epoch)

            # train for one epoch
            self.train(train_dataloader, epoch)

            # evaluate on validation set after 1 epoch
            self.validate(valid_dataloader, epoch)

    # TODO: test function
    def evaluate(self, args):

      # datasets and dataloader for test mode
      self.test_dataset = ImageFolder(self.test_dir, transform=self.transform)

      self.test_dataloader = DataLoader(self.test_dataset,
                                        batch_size=self.test_batch_size,
                                        shuffle=False)


