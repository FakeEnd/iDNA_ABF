# iDNA_ABF

> important! Very sorry, some data in the additional file is wrong due to table format problem. We update the new version in the ./additional_file.pdf

## Introduction

This repository contains code for "iDNA-ABF: a deep learning sequence modeling framework for DNA methylation prediction".

Here is the [article link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02780-1) and [webserver link](https://server.wei-group.net/idnaabf)

We will provide a google drive for download dataset and model parameters in the future. 

Any question is welcomed to be asked by issue and I will try my best to solve your problems.

## Get Started

Thanks to Yingying Yu (She used to be a member of Weilab and now continues her phd life in CityU). She offers a nni version based on pytorchlighting and you can reproduce relevant results by her [repository](https://github.com/YUYING07/iDNA-ABF-automl).

>A tip: the human cell line dataset is relative easy to compare with and outperform to.

### basic dictionary
You can change parameters in `configuration/config.py` to train models.

You can change model structure in `model/ClassificationDNAbert.py` to train models.

You can change training process and dataset process in `frame/ModelManager.py` and `frame/DataManager.py` .

Besides, dataset in paper "iDNA_ABF" is also included in `data/DNA_MS`.

### pretrain model
You should download pretrain model from relevant github repository.

For example, if you want to use [DNAbert](https://github.com/jerryji1993/DNABERT), you need to put them into the pretrain folder and rename the relevant choice in the model.

### Usage

``python main/train.py``
