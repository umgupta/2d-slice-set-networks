## Code for "Improved Brain Age Estimation with Slice-based Set Networks", ISBI 2021

Umang Gupta, Pradeep Lam, Greg Ver Steeg, and Paul Thompson. “Improved Brain Age Estimation
with Slice-based Set Networks.” In: IEEE International Symposium on Biomedical Imaging (ISBI).
2021 (To appear).

To cite the paper, please use the following BibTeX:
```
@article{gupta2021improved,
      title={Improved Brain Age Estimation with Slice-based Set Networks},
      author={Umang Gupta and Pradeep Lam and Greg Ver Steeg and Paul Thompson},
      year={2021},
      eprint={2102.04438},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

![](arch.png)

We proposed a new architecture for BrainAGE prediction, which works by
encoding a single 2D slice in an MRI with a deep 2D-CNN model and
combining the information from these 2D-slice encodings by using set networks
or permutation invariant layers.
Experiments on the BrainAGE prediction problem,
using the UK Biobank dataset showed that the model with the permutation
invariant layers trains faster and provides better predictions compared
to the other state-of-the-art approaches.

## Running the code

### Package requirements
The code is tested with python 3.8, but it should work with python 3.7 or higher.
See `requirements.txt` for the package requirements.

### Data requirements
We have used UKBB MRI scans for training. The  final  dimension of the  images is 91×109×91 and they are loaded via nibabel. Our code requires the csv
files of train/test/valid dataset. See `data` folder for how to set up the csv files. For more details about data and training setup, see our paper.

### Training and evaluation
See `src/shell` folder to reproduce the results in the paper.
The commands for training models with full data, less data and with slicing along different dimensions  is in `table[1,4,5].sh`.
The code to evaluate with missing frame is in `table[2,3].sh`.

### Code and other details
See `config/config.py` and `src/scripts/main.py` to run change or modify params for training or evaluation
Our proposed architecture is in `src/arch/brain_age_slice_set.py`