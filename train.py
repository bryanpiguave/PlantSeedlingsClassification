import pathlib
import tensorflow as tf
import argparse
import configparser
from src.models import model


data_dir="/home/inari/PlantSeedlingsClassification"
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180



train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)



model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
