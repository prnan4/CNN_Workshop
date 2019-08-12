## Training CNN Model

Libraries to import:

```python
import cv2
import time
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
```

Defining path to training and validation images:

```python
train_path = '/Users/nandhinee_pr/CNN_Session/train'
valid_path = '/Users/nandhinee_pr/CNN_Session/valid'
```

Model architecture:

```python
model = Sequential()
model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(28,28, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Convolution2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))

model.add(Dense(2, activation='softmax'))
```

Compiling the model:

```python
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
```

To view the architecture:

```python
model.summary()
```

Image augmetation:

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(28,28),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_path,
        target_size=(28,28),
        batch_size=32,
        class_mode='categorical')
```

For tensorboard visualisation:

```python
NAME = "Cats-vs-dogs-CNN"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
keras.backend.get_session().run(tf.global_variables_initializer())
```

For training the model:

```python
model.fit_generator(
        train_generator,
        samples_per_epoch=19998,
        nb_epoch=10,
        validation_data=validation_generator,
        nb_val_samples=4998,
        callbacks=[tensorboard])
```

To save the model:

```python
model.save('/Users/nandhinee_pr/CNN_Session/model1.h5')
train_generator.class_indices
```