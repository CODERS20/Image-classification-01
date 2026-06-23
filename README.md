# ODYSSEY x Scale Project

## Training Outcome
* **Final Test Accuracy:** 74.509%
* **Precision (Weighted):** 76.12%
* **Recall (Weighted):** 73.20%
* **F1-Score (Weighted):** 73.19%

---

## Confusion Matrix
![Confusion Matrix](download2.png)

---

## Dataset Sources Used
A total of 1,522 images were used to train and test the model. The dataset is evenly split across 5 categories.
* **(https://noirlab.edu/public/images/archive/category/starclusters/page/2/?sort=-release_date)**
* **(https://www.kaggle.com/datasets/akhileshravi/nebula-images)**
* **(https://www.kaggle.com/datasets/robertmifsud/resized-reduced-gz2-images)**
* **(https://universe.roboflow.com/fashion-8zzww/planet-2vlvi/dataset/2)**
* **(https://esahubble.org/)**

---

## Model Architecture 
* **Core Network:** Manually built the standard ResNet structural bottleneck layers. These let the network learn deeper, complex details without running into vanishing gradient problems.
* **Dimensional Alignment of Tensors:** When the data moves between deep stages, its image grid size shrinks and its channel depth expands. Used a custom 1x1 2D Convolutional shortcut path (identity_downsample) to resize the tensors automatically so they match perfectly for matrix addition.
* **Final Output:** At the very end, a single linear layer is used (nn.Linear(2048 -> 5)) to map the network's high-level features down to our 5 final celestial classes.

---

## Setup

1. Clone this project repository and satisfy the local environment library stack requirements:
   ```bash
   pip install -r requirements.txt
