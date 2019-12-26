## Santoro et al. A simple neural network module for relational reasoning. NIPS, 2017. (in PyTorch)
## Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI, 2018 (in PyTorch)

## Description

This repository implements and reproduces two VQA models on Sort-of-CLEVR:
- Relation Networks (RNs)
- FiLM


## Get Started

To use this repo, please download the dataset (Sort-of-CLEVR) below "/data" directory


## Train/Eval

**Train**
- To train RNs
```Shell
./scripts/train_RNs.sh
```
- To train FiLM
```Shell
./scripts/train_FiLM.sh
```
**Eval**

For evaluation, trained model-weights should be located at "./models/here".

- To evaluate RNs
```Shell
./scripts/eval_RNs.sh
```
- To evaluate FiLM
```Shell
./scripts/eval_FiLM.sh
```


## Benchmark (Sort-of-CLEVR)

Below is benchmark result. For fair comparison on relational reasoning ability, I used same CNN visual feature extractor for both models. 

|  models | #param | Accuracy (relational qst)[%] | Accuracy (non-relational qst)[%] |
|:------: | :----: |:----:  | :----:  |
| RNs     |  1.38M |   92   |   99    | 
| FiLM    |  1.72M |   94   |   99    |


## Contact

- Ho-Deok Jang
- Email: jhodeok@gmail.ac.kr
- Homepage: https://sites.google.com/view/hdjangcv
