# Team Project 2: Predicting Human Preferences for LLM Response Enhancement

Teammates: 
• XU LINRUI, 50251600 
• HUANG FANRU, 20214788 
• FANG JINGYI, 20223178 
• FEI XIZE, 20212288 

## Overview
This project addresses the challenging task of predicting human preferences between Large Language Model (LLM) responses, a crucial problem in AI alignment and model evaluation. The competition dataset from Kaggle contains 57,485 training samples and approximately 25,000 test samples, where each instance presents a prompt alongside two alternative responses (A and B) from different models, with human annotations indicating the preferred response or a tie.

Our approach follows a progressive methodology, beginning with simple baseline models and advancing through increasingly sophisticated techniques including sentence embeddings, feature engineering, ensemble methods, and parameter-efficient fine-tuning. We achieved a leaderboard score of **1.02621** on Kaggle, demonstrating consistent improvement through each development phase. 



This repository indexes all materials for convenient reproduction. The project was developed by Team 8 as part of the CS 53744 Machine Learning Project course.

## Reproduction Instructions
To reproduce our results, follow these steps:

1. **Environment Setup**: Use Kaggle Notebooks with Python 3.10 and standard machine learning libraries (e.g., transformers, lightgbm, scikit-learn). All experiments were conducted with fixed random seeds (42) for consistency.

2. **Code Execution**: 
   - Load `Team8_Final_Code.ipynb` into a Kaggle competition Notebook (e.g., for the "LLM Prompt Recovery" competition).
   - Run the code in the **Final Model** section, which consists of two main parts:
     - **Part 1**: This segment produces our highest achieved score (leaderboard score 1.02621). It leverages an ensemble approach without fine-tuning, combining embeddings and bias-aware features.
     - **Part 2**: This segment implements our ideal model structure using fine-tuned DeBERTa with LoRA, but in practice, it performs slightly worse than Part 1 due to potential information loss during fine-tuning. Use this for experimental comparison.

3. **Runtime**: The complete pipeline takes approximately 2 hours on P100 GPU hardware provided by Kaggle. Ensure sufficient resources are allocated.

## Model Architecture
Our final model architecture integrates optimized components for robust preference prediction. It employs LoRA fine-tuning of DeBERTa-v3-small to generate task-specific document embeddings, applies PCA for dimensionality reduction, and concatenates them with handcrafted bias-aware features (e.g., length differentials, lexical diversity ratios, cosine similarity). This enriched feature set is fed into a 5-fold LightGBM classifier to mitigate biases and ensure generalization.



## Key Results
- **Best Validation Log Loss**: 1.02832 (Final Model)
- **Kaggle Leaderboard Score**: 1.02621
- Performance comparison across models:
  - Baseline1 (BoW+LR): 1.2867
  - Baseline2 (MiniLM+LR): 1.0850
  - Final Model: 1.02832

Error analysis revealed that bias-aware features effectively mitigate positional bias, though tie prediction remains challenging. 



## Limitations and Notes
- The dataset may contain inherent biases that propagate to models.
- Computational constraints limited hyperparameter tuning; future work could explore larger architectures.
- For full details, refer to the project documentation in this repository.

## Contact
For questions, please refer to the repository issues or contact Team 8 members: XU LINRUI, HUANG FANRU, FANG JINGYI, FEI XIZE.
