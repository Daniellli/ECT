<!--
 * @Author: daniel
 * @Date: 2022-05-19 22:27:52
 * @LastEditTime: 2024-02-15 21:34:51
 * @LastEditors: daniel
 * @Description: 
 * @FilePath: /Cerberus-main/readme.md
 * have a nice day
-->
# ECT: Fine-grained Edge Detection with Learned Cause Tokens [[Paper](https://arxiv.org/abs/2308.03092)]


## Introduction

In this study, we tackle the challenging fine-grained edge detection task, which refers to predicting specific edges caused by reflectance, illumination, normal, and depth changes, respectively. Prior methods exploit multi-scale convolutional networks, which are limited in three aspects: (1) Convolutions are local operators while identifying the cause of edge formation requires looking at far away pixels. (2) Priors specific to edge cause are fixed in prediction heads. (3) Using separate networks for generic and fine-grained edge detection, and the constraint between them may be violated. To address these three issues, we propose a two-stage transformer-based network sequentially predicting generic edges and fine-grained edges, which has a global receptive field thanks to the attention mechanism. The prior knowledge
of edge causes is formulated as four learnable cause tokens in a cause-aware decoder design. Furthermore, to encourage the consistency between generic edges and fine-grained edges, an edge aggregation and alignment loss is exploited. We evaluate our method on the public benchmark BSDS-RIND and several newly derived benchmarks, and achieve new state-of-the-art results. Our code, data, and
models will be made public.

![main](imgs/main.png)



## Quantitative comparison for REs, IEs, NEs, DEs, and Average on BSDS-RIND.

| Method  | Reflectance (ODS) | Reflectance (OIS) | Reflectance (AP) | Illumination (ODS) | Illumination (OIS) | Illumination (AP) | Normal (ODS) | Normal (OIS) | Normal (AP) | Depth (ODS) | Depth (OIS) | Depth (AP) | Average (ODS) | Average (OIS) | Average (AP) |
| :------ | :---------------- | :---------------- | :--------------- | :----------------- | :----------------- | :---------------- | :----------- | :----------- | :---------- | :---------- | :---------- | :--------- | :------------ | :------------ | :----------- |
| HED     | 0.412             | 0.466             | 0.343            | 0.256              | 0.290              | 0.167             | 0.457        | 0.505        | 0.395       | 0.644       | 0.679       | 0.667      | 0.442         | 0.485         | 0.393        |
| RCF     | 0.429             | 0.448             | 0.351            | 0.257              | 0.283              | 0.173             | 0.444        | 0.503        | 0.362       | 0.648       | 0.679       | 0.659      | 0.445         | 0.478         | 0.386        |
| DFF     | 0.447             | 0.495             | 0.324            | 0.290              | 0.337              | 0.151             | 0.479        | 0.512        | 0.352       | 0.674       | 0.699       | 0.626      | 0.473         | 0.511         | 0.363        |
| RINDNet | 0.478             | 0.521             | 0.414            | 0.280              | 0.337              | 0.168             | 0.489        | 0.522        | 0.440       | 0.697       | 0.724       | 0.705      | 0.486         | 0.526         | 0.432        |
| EDTER   | 0.496             | 0.552             | 0.440            | 0.341              | 0.363              | 0.222             | 0.513        | 0.557        | 0.459       | **0.703**   | 0.733       | 0.695      | 0.513         | 0.551         | 0.454        |
| ECT     | **0.520**         | **0.567**         | **0.470**        | **0.371**          | **0.399**          | **0.318**         | **0.516**    | **0.558**    | **0.473**   | 0.699       | **0.734**   | **0.722**  | **0.526**     | **0.564**     | **0.496**    |




![qual](imgs/qualitative.png)











## Table: Quantitative comparison for REs (IIW), IEs (ISTD), IEs (SBU), NEs (NYUDv2), DEs (NYUDv2). The values highlighted in bold denote the best performance in each category.


| Method  | Reflectance (IIW) | Illumination (ISTD ODS) | Illumination (ISTD OIS) | Illumination (ISTD AP) | Illumination (SBU ODS) | Illumination (SBU OIS) | Illumination (SBU AP) | Normal (NYUDv2 ODS) | Normal (NYUDv2 OIS) | Normal (NYUDv2 AP) | Depth (NYUDv2 ODS) | Depth (NYUDv2 OIS) | Depth (NYUDv2 AP) |
| :------ | :---------------- | :---------------------- | :---------------------- | :--------------------- | :--------------------- | :--------------------- | :-------------------- | :------------------ | :------------------ | :----------------- | :----------------- | :----------------- | :---------------- |
| HED     | 0.638             | 0.508                   | 0.515                   | 0.499                  | 0.566                  | 0.618                  | 0.565                 | 0.332               | 0.342               | 0.149              | 0.360              | 0.376              | 0.185             |
| RCF     | 0.594             | 0.492                   | 0.510                   | 0.463                  | 0.535                  | 0.586                  | 0.510                 | 0.320               | 0.325               | 0.120              | 0.347              | 0.364              | 0.172             |
| DFF     | 0.481             | 0.478                   | 0.495                   | 0.299                  | 0.475                  | 0.483                  | 0.297                 | 0.271               | 0.272               | 0.081              | 0.340              | 0.348              | 0.142             |
| RINDNet | 0.519             | 0.547                   | 0.584                   | 0.465                  | 0.557                  | 0.595                  | 0.471                 | 0.333               | 0.337               | 0.156              | 0.357              | 0.369              | 0.175             |
| EDTER   | 0.458             | 0.552                   | 0.631                   | 0.511                  | 0.599                  | 0.651                  | 0.534                 | 0.333               | 0.340               | 0.131              | 0.349              | 0.360              | 0.170             |
| ECT    | **0.641**         | **0.642**               | **0.689**               | **0.664**              | **0.591**              | **0.656**              | **0.599**             | **0.343**           | **0.352**           | 0.146              | **0.369**          | **0.383**          | **0.197**         |



![qual](imgs/transferability_qualitative.png)





## Installation

### Environment 


```
bash install.sh
```
After this finishing this script, the envirnment name, ect, will be created. Then, you can use `conda activate ect` to activate the environment.

### Data preparation

BSDS-RIND dataset ( [BaiDuNetDisk](https://pan.baidu.com/s/1wrxQyqAJQG1adyk4RzGDmw): code qc62) should have the following hierachy inside project_root/data:


```
BSDS-RIND
├── BSDS-RIND
│   └── Augmentation
│       ├── Aug_HDF5EdgeOriLabel
│       └── Aug_JPEGImages
├── BSDS-RIND-Edge
│   └── Augmentation
│       ├── Aug_HDF5EdgeOriLabel
│       └── Aug_JPEGImages
├── test
├── testgt
│   ├── all_edges
│   ├── depth
│   ├── illumination
│   ├── normal
│   └── reflectance
```


### Pretrained model 

The alignment network ([BaiDuNetDisk](https://pan.baidu.com/s/1K_HWsIJOoGrtcOmtcEj9wg), code eka2; [GoogleNetDist](https://drive.google.com/file/d/1DAcgByeyN7zK7Sz-jlDEZeNqwH8dofwH/view?usp=drive_link)) and : should be placed inside `pretrained_models` directory.(



## Training and evaluation 


To train a ECT on BSDS-RIND, run: 

```
bash scripts/train_ect.sh

```

After finishing training, the training and evaluation results can be found at `networks/2023-05-26-XXX`. 


## Reproducing the results reported in our paper

download our trained model ([[BaiDuNetDisk](https://pan.baidu.com/s/1A4okqEcx8VxUE36QpNwupw), code: t6hc; [GoogleNetDisk](https://drive.google.com/file/d/10b8fqd2cea-Ca2dPYXSkD_mLmgWxK5si/view?usp=drive_link)) the evaluate by following steps: 
1. modify the variable of  `resume_model` in `scripts/test_ect.sh` 
<!-- 2. change the `from model.ECT import EdgeCerberusMultiClass`  to `from model.edge_model import EdgeCerberus` -->
2. run: 
```
bash scripts/test_ect.sh
```





<!-- after finishing training, the results is constructed as follows: -->
<!-- 
```
networks
├── 2023-05-26-XXX
    ├──XXX
        ├── eval_res.json
        ├── all_edges
        │   └── met
        ├── attention
        ├── depth
        │   ├── met
        │   ├── modelname-depth.jpg
        │   ├── nms
        │   └── nms-eval
        ├── eval_res.json
        ├── illumination
        │   ├── met
        │   ├── modelname-illumination.jpg
        │   ├── nms
        │   └── nms-eval
        ├── normal
        │   ├── met
        │   ├── modelname-normal.jpg
        │   ├── nms
        │   └── nms-eval
        └── reflectance
            ├── met
            ├── modelname-reflectance.jpg
            ├── nms
            └── nms-eval
    ├──checkpoints
        ├──ckpt_ep0XXX.pth.tar
        ................................................................
...
``` -->


## Transferability Experiments 

For the details of this benchmark please refer to [Transferability Experiments](./transferibility/readme.md)




## Citation

If you find our work useful in your research, please consider citing:

```
@misc{xu2023ect,
      title={ECT: Fine-grained Edge Detection with Learned Cause Tokens}, 
      author={Shaocong Xu and Xiaoxue Chen and Yuhang Zheng and Guyue Zhou and Yurong Chen and Hongbin Zha and Hao Zhao},
      year={2023},
      eprint={2308.03092},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
