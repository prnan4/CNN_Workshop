## Running predictions on the model

Libraries to import:

```python
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

Loading the model:

```python
model = load_model('/Users/nandhinee_pr/CNN_Session/model1.h5')
```

Reading the test image:

```python
img = cv2.imread('/Users/nandhinee_pr/Downloads/dogs-vs-cats/test1/21.jpg')
plt.imshow(img)
```

Function to predict class of the image:

```python
def predict(img):
    ar = np.array(img).reshape((28,28,3))
    ar = np.expand_dims(ar, axis=0)
    prediction = model.predict(ar)[0]
    if prediction[0] == 1:
        return "cat"
    else:
        return "dog"
```

Testing for the image:

```python
img = cv2.resize(img, (28,28))
res = predict(img)
print(res)
```
