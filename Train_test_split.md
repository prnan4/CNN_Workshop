## Training testing data split

To count the number of files in a directory:

```python
import os
import cv2
path = '/Users/nandhinee_pr/CNN_Session/train/cats'
print(len(os.listdir(path)))
```

Training testing data split:

```python
def write(old_path, new_path):
    img = cv2.imread(old_path)
    cv2.imwrite(new_path, img)

path = '/Users/nandhinee_pr/Downloads/dogs-vs-cats/train'
new = '/Users/nandhinee_pr/CNN_Session/train/dogs'
new1 = '/Users/nandhinee_pr/CNN_Session/valid/dogs'

lim = 0
c=0
for filename in os.listdir(path):
    if filename[0]=='d':
        try:
            old_path = path + '/' + filename
            if lim < 9999:
                new_path = new + '/' + filename
                write(old_path, new_path)
            else:
                if c<2499:
                    new_path1 = new1 + '/' + filename
                    write(old_path, new_path1)
                c+=1
            lim = lim+1
        except:
            pass

#follow for cats
```