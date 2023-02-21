## Code for training Deep Neural networks on Brain MRIs

This repository contains codes for two papers:
- "Improved Brain Age Estimation with Slice-based Set Networks", ISBI 2021 [[Link](https://arxiv.org/abs/2102.04438)]
- "Transferring Models Trained on Natural Images to 3D MRI via Position Encoded Slice Models", ISBI 2023 [Link coming soon]

### Citation

To cite the paper, please use the following BibTeX:

```
@inproceedings{gupta2021improved,
  title={Improved brain age estimation with slice-based set networks},
  author={Gupta, Umang and Lam, Pradeep K and Ver Steeg, Greg and Thompson, Paul M},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages={840--844},
  year={2021},
  organization={IEEE}
}
```

```Bibtex for ISBI 2023 manuscript coming soon```

![](arch.png)
We proposed a new architecture for making predictions from 3D MRIs, which works by encoding each 2D slice in an MRI with a deep 2D-CNN model and combining the information from these 2D-slice encodings by using set networks or permutation invariant layers or operations such as mean. In the ISBI 2021 paper, we performed experiments on brain age prediction using the UK Biobank dataset. We showed that the model with the permutation invariant aggregation layers trains faster and provides better predictions than the other state-of-the-art approaches. 

In the ISBI 2023 paper, we introduce positional encodings to incorporate spatial information about the ordering of the slices; and also used ImageNet pretrained ResNet models as the slice encoders. In this paper, we perform experiments on brain age prediction using the UK Biobank dataset and Alzheimer's disease diagnosis on the ADNI dataset. 

## Running the code

### Package requirements

The code is tested with python 3.8, but it should work with python 3.7 or
higher.
See `requirements.txt` for the package requirements.

### Data requirements

We have used UK Biobank & ADNI MRI scans for training. The final dimension of the images is 91×109×91 and they are loaded via nibabel. Our code requires the csv files of train/test/valid dataset. See `data` folder for how to set up the csv files. For more details about data and training setup, see our paper.

### Training and evaluation

See `src/shell` folder to reproduce the results in the paper.
We have organized the commands according to the tables in the paper.

### Code and other details

See `config/` and `src/scripts/main.py` to run change or modify params
for training or evaluation. 
Our proposed architecture is in `src/arch/ukbb/brain_age_slice_set.py`