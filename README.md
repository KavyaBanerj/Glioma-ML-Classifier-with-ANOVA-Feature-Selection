# Glioma-ML-Classifier-with-ANOVA-Feature-Selection

## Project Overview

This project focuses on classifying glioma subtypes using data from The Cancer Genome Atlas (TCGA). The primary goal is to build machine learning models that can predict glioma classes based on the available metadata and processed gene expression data.

## Directory Structure

The project is organized into several directories:

1. **scr/**: Contains scripts and Jupyter notebooks for data preprocessing, model evaluation, and deployment.
   - `model_evaluation.ipynb`: Jupyter notebook for evaluating machine learning models. It contains code for data loading, preprocessing, model training, evaluation, and feature importance analysis.
   - `preprocess_data_labels.ipynb`: Jupyter notebook for preprocessing the data and creating labels.
   - `streamlit_app.py`: Python script for deploying the model using Streamlit.
   - `get_data.sh`: Shell script to download required data from the GDC using the `gdc-client` tool and the manifest file. The data is downloaded into the `data` folder.

2. **models/**: Stores the pre-trained models and scaler.
   - `scaler.pkl`: Saved pre-processing scaler used for data normalization.
   - `xgb_model.pkl`: Pre-trained XGBoost model for classification.

3. **processed/**: Contains processed data files used for training and testing models.
   - `data.csv`: Processed gene expression data.
   - `glioma_labels.csv`: Labels for glioma classification.

4. **metadata/**: Contains metadata related to the samples.
   - `clinical.tsv`: Clinical metadata associated with TCGA samples.
   - `gdc_sample_sheet.tsv`: Sample sheet downloaded from GDC.

5. **root**:
   - `gdc_manifest.txt`: Manifest file listing TCGA files used in this project.

## Models and Processing

1. **Data Preprocessing**:
   - The notebook begins by loading the gene expression data and glioma labels.
   - Genes with no variance across samples are removed.
   - Missing values are checked, and the data is normalized before transposing for model input.
   - Label distributions are checked to ensure class balance.

2. **Model Training and Evaluation**:
   - **Random Forest**: A Random Forest classifier is trained, and feature importance is visualized.
   - **XGBoost Classifier**: An XGBoost model is also trained, which shows slightly better performance than Random Forest, although both models struggle with distinguishing between "Oligodendroglioma" and "Astrocytoma".
   - Models are evaluated using cross-validation and AUC-ROC scores.
   - Feature importance is plotted for both models.

3. **Feature Selection**:
   - Non-parametric ANOVA-based feature selection is used to reduce the dimensionality of the data before training the models.

4. **Saving Models**:
   - The trained XGBoost model and scaler are saved as `.pkl` files for later use.

## Setup Instructions

1. **Install Dependencies**: Use the provided `environment.yml` file to create a conda environment with the necessary dependencies:
   ```bash
   conda env create -f scr/environment.yml
   conda activate tcga_ml
   ```

2. **Data Download**: Run the `get_data.sh` script to download the data from GDC using the `gdc-client`:
   ```bash
   bash scr/get_data.sh
   ```

3. **Data Preprocessing**: Run the `preprocess_data_labels.ipynb` notebook to process raw gene expression data and generate labels for classification.

4. **Model Training and Evaluation**: Use the `model_evaluation.ipynb` notebook to evaluate the pre-trained model or train a new one on the processed data.

5. **Model Deployment**: The project includes a Streamlit app (`streamlit_app.py`) for deploying the model. You can run the app locally using:
   ```bash
   streamlit run scr/streamlit_app.py
   ```

## Usage

1. **Data Preprocessing**:
   - The `preprocess_data_labels.ipynb` notebook will guide you through cleaning and preparing the dataset.

2. **Model Evaluation**:
   - The `model_evaluation.ipynb` notebook includes sections to evaluate the accuracy and other metrics of the model.

## Acknowledgments

- The Cancer Genome Atlas (TCGA) for providing the datasets used in this project.
