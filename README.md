
# Data-Driven Prediction of Protein Unfolding Kinetics

This project aims to leverage existing biological knowledge of protein folding and unfolding dynamics by utilizing data from MechanoProDB, a web-based database for exploring the mechanical properties of proteins. The approach combines biophysical data obtained from single-molecule force spectroscopy experiments with advanced Machine Learning techniques, primarily consisting of transformer-based models. Among these are ESM2 with 8M parameters, and the combination of ProtBERT, pre-trained on 2.1B protein sequences, with SciBERT, which incorporates domain-specific language collected from 1.14M scientific papers. The research focuses on how experimental conditions, including pH, temperature, mechanical forces and denaturants, interact with different types of proteins to influence unfolding behavior and kinetic changes in proteins’ stability. By learning these relationships, this project seeks to build the optimal model architecture capable of predicting mechanical properties of proteins under different scenarios. This is a novel method, enabling scientists to get approximate kinetic parameters and protein unfolding spectra, based on their specific conditions, not requiring laboratory experiments.


## Project Structure

```bash
Capstone/
├── Data/                                    # Original MechanoProDB, ProThermDB
│   ├── MechanoProDB.xlsx
│   └── ProThermDB.tsv
├── Models_Artifacts_mechano/          #Evaluation Metrics + Artifacts (MechanoProDB)
│   ├── MultiInputBiLSTM_all_text_50_epochs_Mean_Pooling/
│   │   ├── metrics_df_lstm.tsv
│   │   ├── numeric_feature_scaler.pkl
│   │   ├── target_scaler.pkl
│   │   └── sci_tokenizer/
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   ├── MultiInputBiLSTM_subset_text_50_epochs_Mean_Pooling/
│   │   ├── LSTM_best_model-001.pth
│   │   ├── metrics_df_lstm.tsv
│   │   ├── numeric_feature_scaler.pkl
│   │   └── target_scaler.pkl
│   │   └── sci_tokenizer/
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   └── BERT_all_text_50_epochs_Mean_Pooling/
│       ├── BERT_best_model.pth
│       ├── metrics_df_bert.tsv
│       ├── numeric_feature_scaler.pkl
│       └── target_scaler.pkl
├── Models_Artifacts_protherm/         # Evaluation Metrics + Artifacts (ProThermDB)
│   └── ESM2_subset_text_50_epochs_Mean_Pooling/
│       ├── ESM_best_model.pth
│       ├── metrics_df_esm.tsv
│       ├── numeric_feature_scaler.pkl
│       ├── target_scaler.pkl
│       └── tokenizer/
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           └── vocab.txt
├── ML_Models/
│   └── metrics_df_tfdf.tsv
│   ├── BERT_all_text_50_epochs_Mean_Pooling/
│   │   ├── BERT_best_model.pth
│   │   ├── metrics_df_bert.tsv
│   │   ├── numeric_feature_scaler.pkl
│   │   ├── target_scaler.pkl
│   │   ├── prot_tokenizer/
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   └── sci_tokenizer/
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   ├── BERT_subset_text_50_epochs_Mean_Pooling/
│   │   ├── metrics_df_bert.tsv
│   │   ├── numeric_feature_scaler.pkl
│   │   ├── target_scaler.pkl
│   │   ├── prot_tokenizer/
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   └── sci_tokenizer/
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   ├── ESM2_all_text_50_epochs_Mean_Pooling/
│   │    ├── ESM_best_model.pth
│   │    ├── metrics_df_esm.tsv
│   │    ├── numeric_feature_scaler.pkl
│   │    ├── target_scaler.pkl
│   │    └── tokenizer/
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   └── ESM2_subset_text_50_epochs_Mean_Pooling/
│       ├── ESM_best_model.pth
│       ├── metrics_df_esm.tsv
│       ├── numeric_feature_scaler.pkl
│       ├── target_scaler.pkl
│       └── tokenizer/
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           └── vocab.txt
├── Scripts/                           #Main Jupyter Notebooks
│   ├── Data_Analysis.ipynb            #EDA for MechanoProDB, ProThermDB
│   ├── Models_notebook.ipynb          #TFDF, ESM-2, ProtBERT, SciBERT, BiLSTM + Visuals 
│   ├── Preprocessing_notebook_MechanoProDB.ipynb   #Tokenization + Regression Imputings 
│   └── Preprocessing_notebook_ProThermDB.ipynb     #Sequence extraction + Tokenizations 
├── Tokenized_results/                 #Preprocessed Tokenized Datasets
│   ├── mechano_train_tokenized_df.pkl
│   ├── mechano_val_tokenized_df.pkl
│   ├── protherm_train_tokenized_df.pkl
│   ├── protherm_val_tokenized_df.pkl
│   ├── tokenized_dataset_none_protbert_scibert_14645_protherm.pkl
│   └── tokenized_dataset_regression_imputings_protbert_scibert_127_mechano.pkl
└── README.md   

```
## Authors

- **Student:** [Diana Tumayan (AUA)](https://www.linkedin.com/in/diana-tumasyan-bb448b265)
- **Supervisor:** Rafayel Petrosyan




## Main Notebooks
This repository includes several key notebooks that support data preparation, model training, and end-to-end usage:



| Notebook (from Scripts/)  | Location     | Description|
| -------- | ------- | ------------------------- |
| `Data_Analysis`| `Scripts` | Check the distributions of values, perform PCA for Dimensionality Reduction, clustering via K-Means, DBSCAN, analysis of feature correlations and importance through Random Forest and Heatmaps. |
| `Models_notebook`| `Scripts` | Preprocessing of original MechanoProDB dataset by tokenizing text features through ProtBERT and SciBERT, applying Regression Imputings for missing numeric values.  |
| `Preprocessing_notebook_MechanoProDB` | `Scripts` | Sequence extraction through queries (PDB/UniProt), preprocessing of original MechanoProDB dataset by tokenizing text features through ProtBERT and SciBERT. |
| `Preprocessing_notebook_ProThermDB`   | `Scripts` | Building models for targets prediction (TFDF, ESM-2, ProtBERT, SciBERT, BiLSTM) , performing Inference and visualizations (Free Energy Profile, Force-Extension Curve). |

#### Get item



## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Contact

For questions, email diana_tumasyan@edu.aua.am.

