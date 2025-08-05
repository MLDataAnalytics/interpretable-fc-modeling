
# Interpretable FC Modeling

This repository contains code and resources for uncovering functional connectivity (FC) patterns predictive of behavioral traits using interpretable machine learning models. The project aims to enhance the generalizability and interpretability of brain-wide association studies (BWAS) by learning fine-grained FC patterns at both regional and participant levels.

## Features

- Interpretable predictive modeling of functional connectivity data
- Joint learning of region-level relevance scores and prediction functions
- Weighted integration of regional predictions for participant-level trait prediction
- Scalable to large neuroimaging datasets
- Evaluation across multiple cohorts for generalizability

## Data Sources

This project uses publicly available neuroimaging datasets:

- **Adolescent Brain Cognitive Development (ABCD) Study**  
- **Human Connectome Project Development (HCP-D)**  

Please refer to the respective data portals for access and data usage policies.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/interpretable-fc-modeling.git
cd interpretable-fc-modeling
pip install -r requirements.txt
