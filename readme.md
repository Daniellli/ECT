# ECT: Fine-grained Edge Detection with Learned Cause Tokens





## Introduction

In this study, we tackle the challenging fine-grained edge detection task, which refers to predicting specific edges caused by reflectance, illumination, normal, and depth changes, respectively. Prior methods exploit multi-scale convolutional networks, which are limited in three aspects: (1) Convolutions are local operators while identifying the cause of edge formation requires looking at far away pixels. (2) Priors specific to edge cause are fixed in prediction heads. (3) Using separate networks for generic and fine-grained edge detection, and the constraint between them may be violated. To address these three issues, we propose a two-stage transformer-based network sequentially predicting generic edges and fine-grained edges, which has a global receptive field thanks to the attention mechanism. The prior knowledge
of edge causes is formulated as four learnable cause tokens in a cause-aware decoder design. Furthermore, to encourage the consistency between generic edges and fine-grained edges, an edge aggregation and alignment loss is exploited. We evaluate our method on the public benchmark BSDS-RIND and several newly derived benchmarks, and achieve new state-of-the-art results. Our code, data, and
models will be made public.

![main](doc/main.png)


![qual](doc/qualitative.png)


## Installation

### Requirements
    
    torch==1.8.1
    torchvision==0.9.1
    opencv-python==4.5.2
    timm==0.4.5

### Data preparation



Then, prepare NYUd2 dataset or your own dataset.

NYUd2 dataset should have the following hierachy:

```
dataset_path
|   info.json
|   train_images.txt
|   train_labels.txt
|   val_iamges.txt
|   val_labels.txt
|
└───image(semantic image folder)
|     └───...
└───gt_sem_40(semantic label folder)
|     └───...
|
|   train_attribute_images.txt
|   train_attribute_labels.txt
|   val_attribute_iamges.txt
|   val_attribute_labels.txt
|
└───attribute(attribute image and label folder)
|     └───aNYU
|           └───...
|
|   train_affordance_images.txt
|   train_affordance_labels.txt
|   val_affordance_iamges.txt
|   val_affordance_labels.txt
|
└───affordance(affordance image and label folder)
      └───Affordance_ground_truth
            └───...
```

#### Attribute

Download prepocessed attribute dataset [HERE](https://drive.google.com/file/d/13s5JUwj8_QFuKGhxElsA4gaIlfll5OEI/view?usp=sharing)
#### Affordance

Download prepocessed affordance dataset [HERE](https://drive.google.com/file/d/1LVR5Og0EQf1z_DoTPfCQt_gVVe46OcYt/view?usp=sharing)
#### Semantic

Download prepocessed semantic dataset [HERE](https://drive.google.com/file/d/1Hg1H37i0QOzNojpgLlh7bx1SgnLlgmNI/view?usp=sharing)

## Run Pre-trained Model

You can download pre-trained Cerberus model [HERE](https://drive.google.com/file/d/1AX_UYa44uW_aPOSykO06GMcfo8mHDRx6/view?usp=sharing).





## Training and evaluating

To train a Cerberus on NYUd2 with a single GPU:




## Citation

If you find our work useful in your research, please consider citing:

```
coming soon 
```