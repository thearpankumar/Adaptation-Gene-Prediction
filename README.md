[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jQbwFb-HLqWTv7pHwN9mDyGylIdVTp4w?usp=sharing)
# Machine Learning for Adaptation Gene Prediction

This project uses machine learning to predict whether a gene from the bacterium *Deinococcus radiodurans* contributes to stress tolerance based on features derived purely from its DNA sequence. The goal is to create a computational tool that can rapidly screen genomes for candidate stress-response genes, potentially aiding in bioengineering and synthetic biology.

## Table of Contents
- [Project Goal](#project-goal)
- [How It Works](#how-it-works)
- [Features Engineered](#features-engineered)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [How to Use This Project](#how-to-use-this-project)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [File Descriptions](#file-descriptions)
- [Example Prediction Workflow](#example-prediction-workflow)
- [Future Work](#future-work)

## Project Goal

The primary objective is to build and train a machine learning model that can classify a given gene as either a "stress-response gene" or a "normal/housekeeping gene." Instead of relying on expensive and time-consuming laboratory experiments, this model leverages patterns in the DNA sequence itself to make predictions.

This serves as a proof-of-concept for a high-throughput screening tool to prioritize genes for further study in newly sequenced organisms, especially extremophiles.

## How It Works

The project follows a classic supervised learning approach:

1.  **Data Collection:** The complete annotated genome of *Deinococcus radiodurans* (`.gbff` format) is downloaded from the NCBI database.
2.  **Labeling:** Genes are assigned one of two labels:
    *   **`Stress` (Positive Class):** A list is created using a hybrid strategy of (a) manually curated, literature-verified stress genes and (b) programmatic searching for functional keywords like "DNA repair," "radiation resistance," "chaperone," etc.
    *   **`Control` (Negative Class):** A list of housekeeping genes is created using a similar strategy, identifying genes for essential functions like "ribosomal protein" or "gyrase."
3.  **Feature Engineering:** Each gene's DNA sequence is converted into a set of meaningful numerical features that the model can understand.
4.  **Model Training:** An XGBoost classifier is trained on the labeled, feature-engineered dataset to learn the patterns that differentiate the two classes.

## Features Engineered

The model uses a rich set of features to build its predictions:

-   **Basic Sequence Properties:**
    -   `Gene Length`: Total length in base pairs.
    -   `GC Content`: Percentage of Guanine and Cytosine bases.
    -   `GC3 Content`: GC content at the third position of codons.
    -   `CG Dinucleotide Frequency`: The relative abundance of CG pairs.
-   **Protein-Level Features:**
    -   `Hydrophobicity (GRAVY)`: The Grand Average of Hydropathicity of the translated protein.
    -   `Isoelectric Point`: The pH at which the translated protein has no net charge.
-   **Pattern-Based Features:**
    -   `Motif Frequencies`: Occurrence of known regulatory motifs.
    -   `K-mer Frequencies`: A high-resolution sequence "fingerprint" based on the frequency of all possible 4-letter DNA substrings (e.g., 'AAGC', 'GATA').

## Machine Learning Pipeline

A robust pipeline ensures the model is trained correctly and its performance is reliable:

1.  **Preprocessing (`VarianceThreshold`):** Automatically removes useless features that are constant across all samples.
2.  **Feature Selection (`SelectKBest`):** Selects the top 100 most informative features using a statistical ANOVA F-test, reducing noise and complexity.
3.  **Handling Class Imbalance (`SMOTE`):** Artificially balances the training data by creating synthetic examples of the rare "Stress" class, preventing the model from becoming biased.
4.  **Hyperparameter Tuning (`GridSearchCV`):** Systematically tests different model configurations (e.g., `learning_rate`, `max_depth`) to find the optimal settings.
5.  **Training (`XGBoost`):** The final model is an XGBoost classifier, a powerful gradient boosting algorithm well-suited for tabular data.

## How to Use This Project

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation
1.  Clone this repository to your local machine:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  Install the required Python libraries using pip:
    ```bash
    pip install biopython scikit-learn pandas xgboost matplotlib imblearn joblib
    ```

### Running the Notebook
1.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
2.  Open the main notebook file (e.g., `gene_prediction_pipeline.ipynb`).
3.  Execute the cells in order from top to bottom. The notebook is self-contained and will automatically:
    - Download the necessary genome data.
    - Perform all data processing and training steps.
    - Save the final model and all required pipeline components.
    - Demonstrate how to load the saved artifacts and make predictions on sample data.

## File Descriptions

```
.
├── gene_prediction_pipeline.ipynb    # Main Jupyter Notebook with all the code.
├── README.md                         # This README file.
└── GCF_000012145.1_ASM1214v1_genomic.gbff # Genome data (downloaded by the notebook).
└── saved_artifacts/
    ├── stress_gene_model.joblib      # The final, trained XGBoost model.
    ├── feature_selector.joblib       # The fitted SelectKBest object.
    ├── variance_thresholder.joblib   # The fitted VarianceThreshold object.
    ├── kmer_vectorizer.joblib        # The fitted CountVectorizer for k-mers.
    └── feature_columns.joblib        # The list of feature names the model expects.
```

## Example Prediction Workflow

After running the main notebook once, you can use the saved artifacts to predict on a new DNA sequence:

1.  **Load Artifacts:** Load the model, selector, thresholder, vectorizer, and column list using `joblib`.
2.  **Process New Sequence:** Apply the exact same feature engineering steps to your new sequence.
3.  **Align Features:** Use `.reindex()` to ensure the new feature vector has the same columns in the same order as the training data.
4.  **Apply Preprocessors:** Transform the data using the loaded `variance_thresholder` and `feature_selector`.
5.  **Predict:** Use `loaded_model.predict()` to get the final classification.

An example of this entire workflow is provided in the final section of the main Jupyter Notebook.

## Future Work
- **Expand the Dataset:** Incorporate data from other extremophiles (e.g., tardigrades, thermophiles) to create a more general and robust model.
- **Explore More Features:** Engineer additional features, such as codon adaptation index (CAI) or predicted protein secondary structure.
- **Try Different Models:** Experiment with other algorithms like LightGBM or deep learning models (e.g., Convolutional Neural Networks) to compare performance.
- **Deploy as a Web App:** Package the model and prediction pipeline into a simple web application where users can paste a DNA sequence and get a prediction.
