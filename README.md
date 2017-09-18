# Hong Kong Flower Identitfication APP
An APP for identify flowers in Hong Kong with Deep Learning technology.

- Flower list provided by [柴娃娃植物網](https://www.facebook.com/groups/cwwHKplant/)
- Image annotated by [Hong Kong Deep Learning](https://www.facebook.com/groups/170776840085989/)

![screen.jpg](/images/screen.jpg)

## Training
Go to `options.py` and change data_dir to your own dataset *ABSOLUTE* path.

Choose which library, model, optimizer and loss to run in `options.py` by changing `self.configs`.

`python train.py` to start training.

## Adding your own model
To add your model, simply do the following:
1. create your model class in `core/YOURLIBRARYCHOICE/models`, note that it must take two arguments `(args, num_classes)`.
2. add your model class to `ModelsDict` in `core/YOURLIBRARYCHOICE/parser.py`
3. add your model, optimizer and loss function of your choice to `CONFIGS` in `options.py`
4. change `self.configs` to your model in `options.py`

## Splitting Dataset
Currently I am trying to split the dataset into 80:10:10 ratio, but some classes have more than 20 jpgs.

0 - 17.jpg       -> train

18.jpg           -> valid

19.jpg or above  -> test

## TODO
### pyTorch:
- [ ] Visdom.
- [ ] logger for saving results.
- [ ] model.test()

### Keras:
- [x] Model.
- [x] dataloading.
- [x] model.train()
- [x] model.validate()
- [ ] model.test()
- [ ] tensorboard.
