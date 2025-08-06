# Interpretable Modeling of Functional Connectivity to Uncover Patterns Predictive of Cognitive Function in Youth


[![bioRxiv](https://img.shields.io/badge/bioRxiv-Preprint-blue.svg)](https://doi.org/10.1101/2025.03.09.642155)

![framework](https://github.com/MLDataAnalytics/interpretable-fc-modeling/blob/main/figures/framework.png?text=Interpretable+FC+Modeling)

*Interpretable predictive modeling of FC-behavior association. Our model learns fine-grained FC patterns predictive of behavioral traits at both regional and participant levels, capturing the overall FC-behavior association. Region-wise predictions are integrated using region-wise relevance scores, yielding a participant level prediction and an interpretable measure of each region’s contribution. These regional predictions and relevance scores are optimized collaboratively using a mean square error (MSE) based loss to enhance prediction performance.*

---

## 🧠 Overview
This repository contains code and resources for uncovering functional connectivity (FC) patterns predictive of behavioral traits using interpretable machine learning models. The project aims to enhance the generalizability and interpretability of brain-wide association studies (BWAS) by learning fine-grained FC patterns at both regional and participant levels.

### Features

- Interpretable predictive modeling of functional connectivity data
- Joint learning of region-level relevance scores and prediction functions
- Weighted integration of regional predictions for participant-level trait prediction
- Scalable to large neuroimaging datasets
- Evaluation across multiple cohorts for generalizability

---

## 📊 Data Sources

This project uses publicly available neuroimaging datasets:

- **Adolescent Brain Cognitive Development (ABCD) Study**  
- **Human Connectome Project Development (HCP-D)**  

Please refer to the respective data portals for access and data usage policies.

---

## 🚀 Getting Started

### System Requirements

-   Python (3.9.18)
-   PyTorch (1.13.1)
-   NumPy (1.25.2)
-   pandas (1.5.3)
-   SciPy (1.11.3)

### Installation

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/MLDataAnalytics/interpretable-fc-modeling.git
```
2.  Navigate to the project directory:
```bash
cd interpretable-fc-modeling
```
3. Install the required dependencies. It's recommended to do this in a virtual environment.
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Model Traing

To train a model, execute the `train_region_interpret_model.py` script with the following arguments:

```bash
python ./src/train_region_interpret_model.py ${fold_id} ${trait_id} ${data_dir} ${train_list} ${fc_mode} ${output_dir} ${validation_list}
```

**Arguments:**

* `fold_id`: Training fold index (e.g., `fold1`, ..., `fold5` for five-fold cross-validation). This is used in the output file names.
* `trait_id`: Target trait index (e.g., `'1'` for `'pc1'`, which represents general cognition in the ABCD study).
* `data_dir`: Path to the directory where the functional connectivity (FC) `.mat` files are saved.
* `train_list`: A `.csv` file containing the FC file names and behavioral traits for the training set. Each row corresponds to one individual.

    **Example Format** (`fc_file`, `cognitive_measure_1`, `cognitive_measure_2`, ...):

    ```csv
    sub-NDARINV5H1AW18P_fc.mat,0.54091746403823,-0.0192227192451836,0.792918923481453
    sub-NDARINVUCUU472H_fc.mat,-1.16157907198747,0.0439085846756245,-0.732536778124967
    sub-NDARINV99NA21PA_fc.mat,1.55319376072347,-0.751235818323214,0.0310268897410735
    ```

    The `.mat` file should contain a variable `fc_p2p` which is the whole-brain FC matrix (e.g., a 352x352 matrix for 333 cortical regions from Gordon atlas and 19 subcortical regions).

* `fc_mode`: The type of functional connectivity to use.
    * `all`: Whole-brain FC (e.g., 352x352).
    * `cortex`: Cortical-to-cortical FC (e.g., 333x333).
    * `subcortical`: Subcortical-to-cortical FC (e.g., 19x333).
* `output_dir`: Path to the directory where the trained model will be saved (e.g., `output_dir/model_fold1_mse_pc1_Adam_regAtt_1.0/weights_200.pth`).
* `validation_list`: A `.csv` file with the same format as `train_list` for the validation set.

### Model Testing

To test the model, execute the `test_region_interpret_model.py` script:

```bash
python ./src/test_region_interpret_model.py ${fold_id} ${trait_id} ${data_dir} ${test_list} ${fc_mode} ${output_dir}
```

**Arguments:**

* `fold_id`: Testing fold index.
* `trait_id`: Target trait index.
* `data_dir`: Path to the directory where the FC files are saved.
* `test_list`: A `.csv` file containing the FC file names for the testing set.
* `fc_mode`: Must be one of `all`, `cortex`, or `subcortical`.
* `output_dir`: Path where testing results will be saved. The script expects the corresponding trained model to be present in a subdirectory (e.g., `output_dir/model_fold1_mse_pc1_Adam_regAtt_1.0`).

---

### 📜 Manuscript & Citation

For a comprehensive understanding of the method, please refer to the full manuscript. If you find this method useful or inspiring in your research, please consider citing:

```bibtex
@article {Li2025.03.09.642155,
	author = {Li, Hongming and Cieslak, Matthew and Salo, Taylor and Shinohara, Russell T. and Oathes, Desmond J. and Davatzikos, Christos and Satterthwaite, Theodore D. and Fan, Yong},
	title = {Uncovering functional connectivity patterns predictive of cognition in youth using interpretable predictive modeling},
	elocation-id = {2025.03.09.642155},
	year = {2025},
	doi = {10.1101/2025.03.09.642155},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/03/10/2025.03.09.642155},
	eprint = {https://www.biorxiv.org/content/early/2025/03/10/2025.03.09.642155.full.pdf},
	journal = {bioRxiv}
}
```
