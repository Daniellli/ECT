<!--
 * @Author: daniel
 * @Date: 2024-02-15 20:57:06
 * @LastEditTime: 2024-02-15 21:32:04
 * @LastEditors: daniel
 * @Description: 
 * @FilePath: /Cerberus-main/transferibility/readme.md
 * have a nice day
-->

# Transferability Experiments
This document aims to detail the benchmark of Transferability Experiments.

![Transferability](imgs/Transferability_qualitative_res.png)

##  SBU 

1. download the [SBU](https://www3.cs.stonybrook.edu/~cvl/dataset.html) dataset and the [pre-processed data](https://drive.google.com/file/d/1_VSZqJp-x_E9gIyHrJwM3-wlBD2eRds7/view?usp=drive_link). 
2. structure the the data as follows:

```
└── data
    └── SBU
        └── SBU-shadow
            ├── ImageSets
            ├── SBU-Test
            │   ├── EdgeMap
            │   ├── EdgeMapMat
            │   ├── ShadowImages
            │   └── ShadowMasks
            └── SBUTrain4KRecoveredSmall
                ├── EdgeMap
                ├── EdgeMapMat
                ├── ShadowImages
                └── ShadowMasks
```
Note: SBUTrain4KRecoveredSmall is optional  as we only use this data for evaluation.


3.  run following script for evaluation
```
python transferibility/test_ISTD_SBU.py --eval_dataset SBU
```

##  ISTD 

1. download [ISTD](https://drive.google.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view) dataset and our [pre-processed data](https://drive.google.com/file/d/1GNxS4rMzff7ZKHg2rcNUPE9VqcpueGK8/view?usp=drive_link).
2. structure the the data as follows:
```
└── data
    └── ISTD
        └── ISTD_Dataset
            ├── ImageSets
            ├── test
            │   ├── EdgeMap
            │   ├── EdgeMapMat
            │   ├── test_A
            │   ├── test_B
            │   └── test_C
            └── train
                ├── EdgeMap
                ├── EdgeMapMat
                ├── train_A
                ├── train_B
                └── train_C
```
Note: train directory is optional  as we only use this data for evaluation.
3.  run following script for evaluation

```
python transferibility/test_ISTD_SBU.py --eval_dataset ISTD
```


## IIW 

1. Downlaod the [IIW dataset](http://opensurfaces.cs.cornell.edu/publications/intrinsic/#download). 
2. structure the the data as follows:
```
└── data
    └── IIW
        ├── data_transforms.py
        ├── iiw_dataset.py
        └── iiw-dataset
            ├── data
            ├── ImageSets
            ├── info.json
            ├── README.md
            ├── test.txt
            ├── train.txt
            └── whdr.py
```

3. run following script for evaluation

```
python transferibility/test_IIW.py
```




## NYUD2

1. download the [dataset](https://drive.google.com/file/d/1QSotKnOaf07Pql53M-_S96yBRvlWirzH/view?usp=drive_link)
2. structure the the data as follows:

```
└── data
    └── nyud2
        └── NYU_origin
            └── ...
```

3. run following script for evaluation


```
python transferibility/test_NYU.py
```