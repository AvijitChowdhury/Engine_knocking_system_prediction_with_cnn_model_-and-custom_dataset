# -*- coding: utf-8 -*-
"""EngineKnockingDetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17JMfzL4XtsuAMzlDtlnFqzgjdI0-NO5A

# 1 Downloading Audio Files
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install --upgrade youtube-dl

!pip install pytube

from pytube import YouTube
import os

yt = YouTube('https://www.youtube.com/watch?v=uRtyPbyomxQ&list=PL9R1Zswn-XPCnpyXQLRPYVJr4BXJUYie8&index=15&ab_channel=AutoRepairGuys')

video = yt.streams.filter(only_audio=True).first()

out_file = video.download(output_path=".")

base, ext = os.path.splitext(out_file)
new_file = base + '.mp3'
os.rename(out_file, new_file)

!mv *.mp3 /content/drive/MyDrive/Engine_knocking_dataset/knocking

"""# Starting project"""

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/MyDrive/"
base_dir = root_dir + '/Engine_knocking_dataset/'

"""# Importing Libraries"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import IPython.display as ipd

audiofile_path = base_dir+'knocking/Engine Knocking.mp3'

"""# Loading Audiofiles"""

y, sr = librosa.load(audiofile_path,
                     duration=2,
                     offset=0)

"""# Preprocessing audio"""

D = librosa.stft(y)
D_harmonic, D_percussive = librosa.decompose.hpss(D)

"""# Examining the result"""

# Pre-compute a global reference power from the input spectrum
rp = np.max(np.abs(D))

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp), y_axis='log')
plt.colorbar()
plt.title('Full spectrogram')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp), y_axis='log')
plt.colorbar()
plt.title('Harmonic spectrogram')


plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Percussive spectrogram')

plt.tight_layout()

ipd.Audio(y,rate=sr)

ipd.Audio(librosa.istft(D_harmonic),rate=sr)

ipd.Audio(librosa.istft(D_percussive),rate=sr)

"""# Preparing plots for output"""

mydpi=150
pix_side=256

plt.figure(figsize=(pix_side/mydpi, pix_side/mydpi))

CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
librosa.display.specshow(CQT,x_axis=None,y_axis=None)
plt.axis('off')

"""###the spectrogram of the percussive content in the same format."""

plt.figure(figsize=(pix_side/mydpi, pix_side/mydpi))

CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(librosa.istft(D_percussive), sr=sr)), ref=np.max)
p=librosa.display.specshow(CQT,x_axis=None,y_axis=None)
plt.axis('off')

"""**save the file**"""

p.figure.savefig('test.png')

"""**opening the file**"""

from IPython.display import Image
Image(filename='test.png')

"""# Creating training dataset"""

!pip install soundfile

class Spectrogram:
  def __init__(self, audiofile_path, dpi=150, side_px=256, total_duration=10, duration=2):
    import numpy as np
    import matplotlib.pyplot as plt

    import librosa
    import librosa.display

    import os

    import soundfile as sf

    filepath, extension = os.path.splitext(audiofile_path)

    slices = int(total_duration / duration)

    for i in range(slices):
      spectrogram_path = filepath + '_' + str(i) + '.png'
      audio_slice_path = filepath + '_' + str(i) + '.wav'
      y, sr = librosa.load(audiofile_path,
                     duration=duration,
                     offset=duration*i)
      sf.write(audio_slice_path,y,sr)
      D = librosa.stft(y)
      D_harmonic, D_percussive = librosa.decompose.hpss(D)
      # Pre-compute a global reference power from the input spectrum
      rp = np.max(np.abs(D))
      plt.figure(figsize=(side_px/dpi, side_px/dpi))

      CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(librosa.istft(D_percussive), sr=sr)), ref=np.max)
      p=librosa.display.specshow(CQT,x_axis=None,y_axis=None)
      plt.axis('off')
      figure = p.figure
      figure.savefig(spectrogram_path)
      plt.close(figure)

"""# New Section"""

import os
dirs = [base_dir+'knocking2/',base_dir+'normal2/']

for dirry in dirs:
  print(dirry)
  for filename in os.listdir(dirry):
    if filename.endswith('.wav'):
      print(filename)
      Spectrogram(dirry+filename)

!curl -s https://course.fast.ai/setup/colab | bash

"""# Importing fastai"""

from fastai.vision import *

classes = ['knocking','normal']

path =Path('/content/gdrive/MyDrive/Engine_knocking_dataset2')

!ls /content/gdrive/MyDrive/Engine_knocking_dataset2

"""**Validating Image Files**"""

for c in classes:
    print(c)
    verify_images(path/c)

!pip install fastai==1.0.58

import numpy as np
from fastai.vision.data import ImageDataBunch

"""# Define Data Object"""

# ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, size=224, num_workers=4).normalize(imagenet_stats).fillna(0)
try:
  np.random.seed(42)
  data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, size=224, num_workers=4).normalize(imagenet_stats)
except:
  print('exception')

data.classes

"""# display Images from dataset"""

data.show_batch(rows=3, figsize=(7,8))

"""# Show image classes and counts"""

data.classes, data.c, len(data.train_ds), len(data.valid_ds)

"""# Fetch resnet34 model"""

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

"""# Train model"""

learn.fit_one_cycle(4)

"""# Save model"""

learn.save('stage-1')

"""# Unfreeze Top Layers"""

learn.unfreeze()

"""# Final Learning Rate"""

learn.lr_find()

"""**Plot Learning Rate**"""

# learn.lr_find(start_lr=1e-5, end_lr=1e-1)
learn.recorder.plot()

"""# Retain Top Layers"""

learn.fit_one_cycle(2, max_lr=slice(4e-6,4e-4))

"""# Save Model"""

learn.save('stage-2')

learn.save('stage-2')

"""# Interpret Model"""

###Load Model

learn.load('stage-2');

"""# Create Interpretition From Learner"""

interp = ClassificationInterpretation.from_learner(learn)

"""# Plot confusion Matrix"""

interp.plot_confusion_matrix()

"""# Plot Top Losses"""

losses,idxs = interp.top_losses(10)

len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_top_losses(9)

"""# Show Audiofile Players for Top Losses"""

import IPython.display as ipd
import os

for img_path in data.valid_ds.items[idxs]:
  filepath, extension = os.path.splitext(img_path)
  audio_slice_path = filepath + '.wav'
  print(filepath)
  ipd.display(ipd.Audio(audio_slice_path))

"""# Export Model"""

learn.export()

learn.export('Model_v1.pkl')

learn

!cp /content/model_v1 /content/gdrive/MyDrive/Engine_knocking_dataset2

