# Adjust the Config. 

Select experiment:
```yaml
experiment: experiment1 #[experiment1, experiment2, experiment3, experiment4a, experiment4b, experiment5]
```

Select run (with different seeds):
```yaml
run: run0 #[run0, run1, run2, run3, run4]
```

Select feature extractor:
```yaml
embedding_network: densenet201 #[densenet201, resnet101]
```

Set paths:
```yaml
io_root: ./outputs

weight_dir: /PATH/TO/WEIGHTS

dataset_dir: /PATH/TO/DATASETS
```

# First, you need to compute the metrics for the training and test data. You can also visualize the IoU estimates obtained by the meta-regressor.

*metaseg_main.py*

```python
if cfg.tasks.metaseg_train:
```
	MetaSeg computes probs (.npy), metrics (.p) and components (.p) for the training data and further saves the prediction (.png) and Ground Truth (.png).

```python
if cfg.tasks.metaseg_test:
```
	MetaSeg computes probs (.npy), metrics (.p) and components (.p) for the test data and further saves the prediction (.png) (and Ground Truth (.png)).

```python
if cfg.tasks.metaseg_visualize:
```
	The predictions of the meta-regression model are visualized for the test data.


# Then, detect anomalies by thresholding on the meta-regression score and save some useful properties.

*compute_embeddings.py*

```python
if cfg.tasks.compute_embeddings:
```
	Detection of the anomalies by thresholding on the meta-regression score. For each image, which contains an anomaly, we save the image path, the name of the segmentation model and the dataset, and the component-wise estimated IoU scores. For each anomaly, we compute a bounding box, which we crop out and pass it to the feature extractor. We save the features, the box coordinates, the component indices, the image from which the patch originates, and finally the most frequent Ground Truth and predicted class with respect to the involved pixels.

# Reduce the feature dimension, clean up the embedding space and perform clustering. Create pseudo-labels for each cluster.

*detect_clusters.py*

```python
if cfg.tasks.detect_clusters:
```
	Dimensionality reduction with PCA + t-SNE is performed after filtering the feature space. We remove anomalies without "class salad", meaning that they are (almost) completely assigned to one class. Then, we look for novel classes in the 2D space using DBSCAN clustering. For each proposed cluster with at least 30 elements, we generate the pseudo-labels.

# To extend the DNN by novel classes, we propose to have a look on the pseudo-labels and delete clusters which are obviously wrong. For the remaining clusters, add a label entry to the respective dataset, e.g.

```python
Label('cluster_1', 34, 17, 'novel', 6, True, False, (255, 102, 0)),
```

for experiment1. 

