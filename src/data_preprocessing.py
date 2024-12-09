import numpy as np
import cv2
import os

def load_dataset(input_dir,categories):

    x=[]
    y=[]

    for category_idx, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)  
        if not os.path.exists(category_path):
            print(f"Directory {category_path} does not exist.")
            continue

        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            img = cv2.imread(img_path)

            if img is not None:  
                img = cv2.resize(img, (224, 224))  
                x.append(img)
                y.append(category_idx)
            else:
                print(f"Failed to load image: {img_path}")

    x = np.asarray(x, dtype=np.float32)/255.0
    y = np.asarray(y, dtype=np.int32).reshape(-1,1)
    return x, y
def one_hot(y,num_categories):
    y_one_hot=np.zeros((y.shape[0],num_categories))
    y_one_hot[np.arange(y.shape[0]),y.flatten()]=1
    return y_one_hot