import os
import numpy as np
from PIL import Image

for im in os.listdir('/home/uhlemeyer/outputs/Cluster_experiment1_run10/human/semantic_id_all'):
    label = np.array(Image.open('/home/uhlemeyer/outputs/Cluster_experiment1_run10/human/semantic_id_all/' + im))
    label_ignore = np.copy(label)
    label_ignore[label < 200] = 0
    Image.fromarray(label_ignore.astype('uint8')).save('/home/uhlemeyer/outputs/Cluster_experiment1_run10/human/semantic_id/' + im)