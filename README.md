# Speech Impediment Detection using Fine-tuned Lightweight LLM and Traditional ML Classifiers

## Overview
This project focuses on developing a multi-modal approach for the classification of speech disorders using both audio features and textual transcriptions. It leverages DistilBERT, a lightweight variant of BERT, and a Feed-Forward Neural Network (FFNN) to extract meaningful embeddings from Automatic Speech Recognition (ASR) transcriptions and Mel-Frequency Cepstral Coefficients (MFCC) along with Chroma features from audio clips. 

## Features
- **Dataset**: Utilizes the Sep-28K dataset and the Dysarthria vs Non-Dysarthria dataset.
- **ASR Model**: Uses OpenAI Whisper for generating transcripts.
- **Feature Extraction**: Extracts MFCC and Chroma features from audio clips.
- **Fine-tuning**: LoRA fine-tuned DistilBERT to generate MFCC embeddings.
- **Machine Learning Models**: Includes Random Forest, SVM, Naive Bayes, XGBoost, and Decision Trees for classification.
- **Novelty**: Integrates audio and text features using LoRA fine-tuning for enhanced classification accuracy.

## Installation
```bash
pip install -r requirements.txt
```

## Repository Structure
```
├── Sep-28K
│   ├── Additional Scripts/ (Scripts for processing data)
│   ├── Datasets/ (Processed dataset for training)
│   ├── Demo-Sep28K.ipynb (Demo notebook for testing models)
│   ├── DistilBert_FFNN.ipynb (Training notebook for models)
│   ├── ffnn_model.pth (Saved FFNN model)
│   ├── Sample Audio Files/
│
├── Dysarthria vs Non-Dysarthria
│   ├── Dataset.zip (Transformed dataset for training)
│   ├── Demo-Dysarthria.ipynb (Demo notebook for testing models)
│   ├── LORA_DistilBert_MFCC.ipynb (Notebook for fine-tuning DistilBERT)
│   ├── Speech Feature Transformation.py (Script for MFCC & Chroma extraction)
│   ├── Sample Audio Files/
```

## Running the Project
### Sep-28K Dataset
1. Open `DistilBert_FFNN.ipynb` and run all cells.
2. Open `Demo-Sep28K.ipynb` and run all cells to test the saved models.

### Dysarthria vs Non-Dysarthria Dataset
1. Unzip `Dataset.zip` and place the `.csv` file in the Dataset folder.
2. Open `LORA_DistilBert_MFCC.ipynb` and run all cells to fine-tune DistilBERT and train ML models.
3. Open `Demo-Dysarthria.ipynb` and run all cells to test the model.

## Results
- **Sep-28K Dataset:** Best accuracy (81.9%) achieved using SVM and Random Forest.
- **Dysarthria vs Non-Dysarthria Dataset:** XGBoost performed best with 93.5% accuracy.

## Future Work
- Explore deeper transformer-based models specifically designed for audio processing.
- Collaborate with medical institutions to acquire more annotated datasets.

## References
Refer to the `References` section in the project report for detailed citations.
