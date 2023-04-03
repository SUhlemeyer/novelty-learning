# Preparation 
All model weights, dataset splits and pseudo-labels that we used in our experiments will be published soon. 
Save the weight and dataset folders into the current working directory. Download the Cityscapes (https://www.cityscapes-dataset.com/) and A2D2 (https://www.a2d2.audi/a2d2/en.html) datasets and adjust the dataset roots in ```config.yaml```. 

Run 
```shell
$ python convert_A2D2_to_Cityscapes.py
``` 
to convert the A2D2 labels to match the Cityscapes IDs. The labels will be saved in ```./datasets/a2d2_label```. (will be published soon)

It is recommended to create a python3 environment and install all required packages (python version 3.6):
```shell
$ python3 -m venv venv
$ source venv/bin/activate

$ pip install --upgrade pip
$ pip install -r requirements.py
```

# Run Extension and Evaluation Code
You can extend the models using our data by running
```shell
$ python main.py ++experiment={experiment} ++tasks.extend_model=True 
```
for the experiments *experiment1, experiment2, experiment3, experiment4a, experiment4b* and *experiment5*.

# Run all the Code

**First, you need to compute the metrics for the training and test data. You can also visualize the IoU estimates obtained by the meta-regressor.**


```shell
$ python main.py ++experiment={experiment} ++tasks.metaseg_train=True 
```
MetaSeg computes probs (.npy), metrics (.p) and components (.p) for the training data and further saves the prediction (.png) and Ground Truth (.png).

```shell
$ python main.py ++experiment={experiment} ++tasks.metaseg_test=True 
```
MetaSeg computes probs (.npy), metrics (.p) and components (.p) for the test data and further saves the prediction (.png) (and Ground Truth (.png)).

```shell
$ python main.py ++experiment={experiment} ++tasks.metaseg_visualize=True 
```
The predictions of the meta-regression model are visualized for the test data.


**Then, detect anomalies by thresholding on the meta-regression score and save some useful properties.**


```shell
$ python main.py ++experiment={experiment} ++tasks.compute_embeddings=True 
```
Detection of the anomalies by thresholding on the meta-regression score. For each image, which contains an anomaly, we save the image path, the name of the segmentation model and the dataset, and the component-wise estimated IoU scores. For each anomaly, we compute a bounding box, which we crop out and pass it to the feature extractor. We save the features, the box coordinates, the component indices, the image from which the patch originates, and finally the most frequent Ground Truth and predicted class with respect to the involved pixels.

**Reduce the feature dimension, clean up the embedding space and perform clustering. Create pseudo-labels for each cluster.**


```shell
$ python main.py ++experiment={experiment} ++tasks.detect_clusters=True 
```
Dimensionality reduction with PCA + t-SNE is performed after filtering the feature space. We remove anomalies without "class salad", meaning that they are (almost) completely assigned to one class. Then, we look for novel classes in the 2D space using DBSCAN clustering. For each proposed cluster with at least 30 elements, we generate the pseudo-labels.

**To extend the DNN by novel classes, we propose to have a look on the pseudo-labels and delete clusters which are obviously wrong. For the remaining clusters, add a label entry to the respective dataset, e.g.**

```python
Label('cluster_1', 24, 17, 'novel', 6, True, False, (255, 102, 0)),
```

for experiment1 (for evaluation reasons, we map the "new" *human* class to the Cityscapes ID for *person*). 

