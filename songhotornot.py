#fastai code copied over from Paperspace
#this code takes thousands of spectrograms of popular (over 100m listens) and unpopular songs to try to work out what makes a great song

!pip install -Uqq timm
! [ -e /content ] && pip install -Uqq fastbook
import timm
import fastbook
import torch
fastbook.setup_book()

from fastbook import *

#uncomment line below if want to see what models are available to train
# timm.list_models()
#timm.list_models(pretrained=True)

from fastai.vision.all import *
path = 'images/spectsv3' #gets all the locally held spectrograms I made

def is_pop(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct = 0.2, seed = 32,
    label_func=is_pop, item_tfms=Resize(712), bs = 16)


learn = vision_learner(dls, resnet50, metrics=error_rate)
learn.fine_tune(6)


#post-training analysis
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.top_losses()
interp.plot_top_losses(5, nrows=5, figsize=(48,48))


#songs that should be popular that aren't and songs that should be unpopular but are
losses, idxs = interp.top_losses(48)
#learn.dls.valid_ds.items
idxs

for i in idxs:
    print(learn.dls.valid_ds.items[i])