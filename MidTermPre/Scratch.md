Mid-Term Presentation
=

CT Image Refinement with Generative Adversarial Networks

## Baseline Method for CT Image Refinement (Res-CNN)

### Dataset

{'train': 17120, 'val': 592} lung photos

with ground-truth image, projection data sampled from 360 angles

we take 72 of them as the source data for reconstruction.

| Methods | RRMSE(train) | RRMSE(val) |
| ---- |:---:| :---:|
| Sparse View FBP | 13.8% | 12.2% |
| Full View FBP   | 6.7%  | 5.5%  |
| Res-CNN(Claimed)| - | 5.1% |

### Method Overview

72 |PRJ| --> 360 --> |FBP| --> image --> |RFN| --> refined-image

### Reproduce

#### 1. Projection Estimation + FBP Network

Details

* Replace momentum with Adam optimizer, 1e-4 lr, 2000 steps pretrain, 3 epochs
* Always use Batch Norm after Conv layers
* Replace ReLU with Leaky-ReLU
* Using Weight sharing
* Periodic Padding

Comparison

* proposed structure

* My implementation

#### 2. Image Refinement Network

Details

* Encoder-Decoder Structure
* Skip Connection

Comparison

* proposed structure /tmp/tmp37eg_n4v/ 0.667

* My implementation /tmp/tmp1oqlm56y/ 0.76

## Using Generative Adversarial Networks