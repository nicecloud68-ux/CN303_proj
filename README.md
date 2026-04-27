# CN303_proj

📩 **Spam Detection System** - A machine learning project for detecting spam messages using multiple classification algorithms.

## About

This repository contains a comprehensive spam detection implementation using various machine learning models. The project analyzes SMS/email messages and classifies them as spam or ham (legitimate) using TF-IDF vectorization combined with engineered features.

## Features

- **Multiple ML Algorithms**: Naive Bayes, Logistic Regression, Random Forest, Gradient Boosting, and SVM
- **Feature Engineering**: URL detection, numeric content analysis, uppercase character counting, punctuation analysis
- **TF-IDF Vectorization**: N-gram based text feature extraction (1-2 grams, max 5000 features)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix analysis
- **Cross-Validation**: 5-fold cross-validation for model robustness assessment

## Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **Random Forest** ⭐ | **0.9839** | **1.0000** | **0.8792** | **0.9357** |
| Gradient Boosting | 0.9803 | 0.9441 | 0.9060 | 0.9247 |
| Logistic Regression | 0.9704 | 0.9203 | 0.8523 | 0.8850 |
| Naive Bayes | 0.9363 | 0.7321 | 0.8255 | 0.7760 |
| SVM | 0.9381 | 0.8636 | 0.6376 | 0.7336 |

**Best Model**: Random Forest with F1-score of 0.9357 and perfect precision (1.0)

## Getting Started

### Prerequisites

- Python 3.7+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib

Install dependencies:
```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nicecloud68-ux/CN303_proj.git
   cd CN303_proj
   ```

2. Ensure you have the `spam.csv` dataset in the project root directory

### Usage

Run the spam detection notebook:
```bash
python spam_detection_complete_notebook.py
```

This will:
1. Load and preprocess the spam dataset
2. Extract features from messages
3. Vectorize text using TF-IDF
4. Train all 5 models
5. Display performance metrics
6. Show confusion matrix for the best model
7. Perform cross-validation analysis

## Project Structure

```
CN303_proj/
├── README.md
├── LICENSE
├── spam.csv                              # Dataset
└── spam_detection_complete_notebook.py   # Main implementation
```

## Dataset Format

The script handles multiple dataset formats:
- Columns: `v1` (label) and `v2` (message)
- Columns: `label` and `message`
- Columns: `category` and `message`

Labels are normalized to: `ham` (0) and `spam` (1)

## Technical Details

### Feature Engineering
- **num_urls**: Count of URLs in message
- **num_numbers**: Count of numeric digits
- **num_exclamation**: Count of exclamation marks
- **num_uppercase**: Count of uppercase letters
- **msg_length**: Total message length

### Text Processing
- Lowercase conversion
- URL replacement with 'url' token
- Digit replacement with 'num' token
- Punctuation removal
- TF-IDF with bigram support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please reach out to the project maintainer.
