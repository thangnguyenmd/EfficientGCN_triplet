import numpy as np
import random
import os
import glob 

label = np.load("data/20220106_VN_trimmed/labels_val.npy")
data = np.load("data/20220106_VN_trimmed/dataset_val.npy")
print(np.unique(label))
pair = zip(label, data)
for i, p in enumerate(pair):
    if i > 100000:
        break
    print(p[0], p[1].shape)
    os.makedirs(os.path.join("data/20220106_VN_trimmed/", "action_" + str(p[0])), exist_ok=True)
    np.save(os.path.join("data/20220106_VN_trimmed/", "action_" + str(p[0]), "a_" + str(i) + ".npy"), p[1])
