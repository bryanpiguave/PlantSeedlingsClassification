import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime 
import hydra
from hydra.core.config_store import ConfigStore
from src.config import SeedlingConfig
import wandb
import tensorflow.compat.v1 as tf1

today=datetime.date.today()
today = today.strftime("%d-%m-%Y")

cs =ConfigStore.instance()
cs.store(name="Seedling_config",node=SeedlingConfig)

@hydra.main(config_path="configs",config_name="train")
def main(cfg:SeedlingConfig):
  data_dir = pathlib.Path(cfg.paths.data_dir)
  

  train_dir = f"{data_dir}/train"
  IMG_SIZE = (cfg.params.img_width, cfg.params.img_height)
  train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    validation_split=0,
    seed=123,
    label_mode='int',
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=cfg.params.batch_size
    )
  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
  ])
  
  IMG_SHAPE = IMG_SIZE + (3,)


  preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
  rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

  base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SHAPE),
                                                include_top=False,
                                                weights='imagenet')
  prediction_layer = tf.keras.layers.Dense(cfg.params.num_classes) 
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

  inputs = tf.keras.Input(shape=(cfg.params.img_width, cfg.params.img_height, 3))
  x = data_augmentation(inputs)
  x = preprocess_input(x)
  x = base_model(x, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)

  model = tf.keras.Model(inputs, outputs)

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= cfg.params.base_learning_rate,
    decay_steps=1000,
    decay_rate=0.9)


  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  wandb.init(project="SeedlingClassification",
    config=tf1.flags.FLAGS, sync_tensorboard=True)


  history = model.fit(train_ds,
                      epochs=cfg.params.initial_epochs,
                      )


  acc = history.history['accuracy']

  loss = history.history['loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.savefig("results/metrics.jpg")

  saved_model_path = f"runs/model_{today}"
  tf.saved_model.save(model, saved_model_path)

if __name__=="__main__":
  main()