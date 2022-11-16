## This project implemented the proposed new architecture Factorized Sum-product Network (FSPN)

## Code structure
The overall structure was inspired and adapted from https://github.com/SPFlow/SPFlow. 
I directly reused some files and functions of this project as well. I thank the SPFlow authors for their excellent code.

This fspn folder contains six sub-folders namely: Algorithm, Data_prepare, Evaluation, Inference, Learning and Structures.
Learning contains the key file: structureLearning.py, which learns the FSPN from scratch using the data.
Structure defines the set of nodes (sum, product, factorize and leaf).

## How to do some simple experiment

Setup the environment using conda (some packages might not support Linux):
```
conda env create -f environment.yml
conda activate fspn
```
Navigate to the Evaluation folder and test the training process of FSPN on some toy datasets.
```
cd fspn/Evaluation
python test_training.py --dataset strong_corr_cat
```
The test_on_GAS.ipynb file provides a concrete example of how to use FSPN for cardinality estimation of a real-world dataset.

## Citation
If you find this project useful please cite our papers:
```
@article{wu2020fspn,
  title={FSPN: A New Class of Probabilistic Graphical Model},
  author={Wu, Ziniu and Zhu, Rong and Pfadler, Andreas and Han, Yuxing and Li, Jiangneng and Qian, Zhengping and Zeng, Kai and Zhou, Jingren},
  journal={arXiv preprint arXiv:2011.09020},
  year={2020}
}
```

```
@article{zhu2020flat,
  title={FLAT: Fast, Lightweight and Accurate Method for Cardinality Estimation},
  author={Zhu, Rong and Wu, Ziniu and Han, Yuxing and Zeng, Kai and Pfadler, Andreas and Qian, Zhengping and Zhou, Jingren and Cui, Bin},
  journal={VLDB},
  year={2021}
}
```
