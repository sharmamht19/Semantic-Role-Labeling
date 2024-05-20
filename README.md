
# Semantic Role Labeling (SRL) for Hindi

<!-- ## Team Information

- **Course:** Introduction to NLP (S24CS7.401)
- **Advisor:** Prof. Manish Shrivastava, Prof. Rahul Mishra
- **Mentor:** Advaith Malladi
- **Team Number:** 54
- **Team Name:** Lang3.1
- **Team Members:**
  - Mohit Sharma (2022201060)
  - Neeraj Asdev (2022201056)
  - Hrishikesh Deshpande (2022201065)
- **Academic Year:** 2023-2024 -->

## Project Description

Semantic Role Labeling (SRL) is a critical task in Natural Language Processing (NLP) aimed at identifying the roles words or phrases play in a sentence. This project focuses on developing an SRL system for Hindi, using both statistical and neural models. The goal is to enhance the performance of SRL systems by accurately labeling arguments in sentences, which is vital for applications like question-answering, inference, and machine translation.

## Why SRL?

1. **Improves NLP Applications:** Enhances tasks such as question-answering, inference, and knowledge graph creation by providing deeper insights into sentence structures and meanings.
2. **Better Translation:** Preserves intended meanings and syntactic structures across languages, leading to more accurate translations.
3. **Accessibility:** Increases linguistic accessibility for diverse communities.
4. **Linguistic Diversity:** Addresses the needs of multilingual communities, particularly in India.

## Datasets

- **Hindi Propbank:** Part of the Hindi-Urdu PropBank project, designed for SRL tasks. It provides a multi-representational and multi-layered Treebank for Hindi-Urdu.
- **Custom Dataset:** Created from the Hindi Propbank, consisting of around 14,000 tokens of Hindi text with head POS, dependency relations, and SRL labels.
- **Additional Dataset:** Collected 1.3k Hindi sentences with arguments, SRL labels, and dependency relations.

## Models and Experiments

### Statistical Models

- **Linear Support Vector Classifier (LinearSVC):**
  - Effective for high-dimensional data.
  - Utilized three different sets of input features capturing various aspects of input data.

### Neural Network Models

1. **FastText Embeddings + BiLSTM Classifier:**
   - Combines FastText embeddings and BiLSTM for non-contextual word representations.
2. **FastText Embeddings + Dependency Relation + BiLSTM Classifier:**
   - Adds dependency relations to enhance syntactic and semantic understanding.
3. **Indic-BERT + MLP Classifier:**
   - Uses Indic-BERT embeddings with a Multilayer Perceptron for contextual word representations.
4. **Indic-BERT + BiLSTM Classifier:**
   - Combines Indic-BERT embeddings with BiLSTM for contextual and sequential modeling.
5. **Indic-BERT + Dependency Relation + BiLSTM Classifier:**
   - Integrates Indic-BERT embeddings and dependency relations for comprehensive SRL.

## Evaluation

- **Baseline Model:** Utilized fundamental features (word, POS tag, argument indicator).
- **Performance Metrics:** Accuracy, precision, recall, and F1-score.
- **Results:**
  - **FastText Embeddings + BiLSTM:** 70.41% accuracy
  - **FastText Embeddings + Dependency Relation + BiLSTM:** 86.86% accuracy
  - **Indic-BERT + MLP Classifier:** 89.81% accuracy
  - **Indic-BERT + BiLSTM Classifier:** 90.02% accuracy
  - **Indic-BERT + Dependency Relation + BiLSTM:** 95.34% accuracy

## Challenges

- Scarcity of suitable datasets for SRL in Hindi.
- Limited resources and references in this domain.
- Need for deeper understanding of Hindi grammar and possibly other Indian languages.
- Small dataset size impacting the reliability of metrics.

## Future Work

- **Exploring Different BERT Layers:** To improve semantic feature representation by testing various BERT layers.

## Directory Structure

```
├── Data
│
├── Code
│   ├── Data Preparation.ipynb
│   ├── NLP_FastText_1.ipynb
│   ├── NLP_FastText_2.ipynb
│   ├── NLP_IndicBert_1.ipynb
│   ├── NLP_IndicBert_2.ipynb
│   ├── NLP_IndicBert_3.ipynb
│   └── NLP_Statistical_Models.ipynb
│
├── Models
│   ├── FastText_BiLSTM_Classifier.pt
│   ├── FastText_BiLSTM_Dep_Classifier.pt
│   ├── IndicBert_BiLSTM_Classifier.pt
│   ├── IndicBert_MLP_Classifier.pt
│   └── IndicBert_MLP_Dep_Classifier.pt
│
├── Research Papers
│   ├── (Paper) Nomani - Towards Building Semantic Role Labeler.pdf
│   └── 28_W29.pdf
│
├── LICENSE
├── Project_PPT.pdf
└── Project_Report.pdf
```

### Directory Details

- **Data:** Contains the datasets used for training and testing the SRL models. These datasets include the Hindi Propbank and additional custom datasets created for this project.

- **Code:** Includes all Jupyter notebooks used for data preparation, model training, and evaluation. Each notebook is named according to its content:
  - `Data Preparation.ipynb`: Scripts for preparing and preprocessing data.
  - `NLP_Statistical_Models.ipynb`: Implementation of statistical models like Linear SVC.
  - `NLP_FastText_1.ipynb`: Initial FastText embedding and model training.
  - `NLP_FastText_2.ipynb`: Further experiments with FastText embeddings.
  - `NLP_IndicBert_1.ipynb`, `NLP_IndicBert_2.ipynb`, `NLP_IndicBert_3.ipynb`: Various experiments using IndicBERT embeddings.
  
- **Models:** Contains the saved models resulting from the training processes. Each model file is named according to the architecture used:
  - `FastText_BiLSTM_Classifier.pt`: Model using FastText embeddings with BiLSTM.
  - `FastText_BiLSTM_Dep_Classifier.pt`: FastText embeddings with BiLSTM and dependency relations.
  - `IndicBert_BiLSTM_Classifier.pt`: IndicBERT embeddings with BiLSTM.
  - `IndicBert_MLP_Classifier.pt`: IndicBERT embeddings with a Multilayer Perceptron (MLP).
  - `IndicBert_MLP_Dep_Classifier.pt`: IndicBERT embeddings with MLP and dependency relations.

- **Research Papers:** Contains relevant research papers and references used during the project:
  - `(Paper) Nomani - Towards Building Semantic Role Labeler.pdf`: A referenced research paper.
  - `28_W29.pdf`: Another referenced research document.

- **LICENSE:** The license file for the project.

- **Project_PPT.pdf:** The presentation slides summarizing the project.

- **Project_Report.pdf:** The detailed project report outlining methodologies, experiments, and results.

## Code and Resources

- **Repository:** [Google Drive](https://drive.google.com/drive/folders/1Bxp186cMNVTH1QoJezUzb2cu0Fxug3De?usp=sharing)

## References

1. Gupta, A., & Shrivastava, M. (2018). Enhancing Semantic Role Labeling in Hindi and Urdu. *LREC 2018 Workshop on Representation Learning for NLP*, 28-32.
2. Anwar, M., & Sharma, D. (2016). Towards Building Semantic Role Labeler for Indian Languages. *Tenth International Conference on Language Resources and Evaluation (LREC'16)*, 4588-4595.
3. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2016). Enriching Word Vectors with Subword Information. *Facebook AI Research*.
