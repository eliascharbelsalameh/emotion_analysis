# inmind.ai Sequential Model

This repository contains the code and documentation for the **inmind.ai Sequential Model** project. The project explores sequential models for natural language processing (NLP), comparing Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) to perform emotion analysis from text data.

## Overview

Sequential data is common in various domains such as time-series analysis, speech recognition, and text generation. Traditional models often treat each data point independently, whereas sequential models capture dependencies over time. This project:
- Explores the limitations of traditional RNNs, including the vanishing and exploding gradient problems.
- Implements LSTM and GRU architectures to better handle long-term dependencies.
- Applies these models to an emotion analysis dataset containing over 420,000 text entries.

## Table of Contents

- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Experiments and Results](#experiments-and-results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Authors](#authors)
- [License](#license)

## Dataset

The emotion analysis dataset consists of:
- **Entries:** 422,746 textual entries.
- **Features:** Each entry includes a sentence and its corresponding categorical emotion label.
- **Preprocessing:** Data cleaning and transformation steps have been applied to remove duplicates and prepare the data for model training.

## Model Architectures

### RNN (Recurrent Neural Network)
- **Concept:** Processes sequential data by maintaining a hidden state that is updated at each time step.
- **Limitation:** Faces challenges with long-term dependencies due to vanishing or exploding gradients.

### LSTM (Long Short-Term Memory)
- **Mechanism:** Uses gates (input, forget, and output) and cell states to effectively capture and maintain long-term dependencies.
- **Performance:** Achieved high performance with precision of 82.52% and F1-score of 82.59% under optimized configurations.

### GRU (Gated Recurrent Unit)
- **Mechanism:** Simplifies the LSTM architecture by combining the forget and input gates into a single update gate, reducing computational complexity.
- **Advantage:** Faster training time compared to LSTM while still addressing gradient issues.

## Experiments and Results

Multiple experiments were conducted with different architectures and hyperparameter settings. Key points include:
- **Embedding:** Utilized word2vec with parameters: `input_size = 300`, `min_count = 3`, and `window = 3`.
- **Hyperparameters:** Explored various configurations for model depth, hidden sizes, learning rates (lr = 0.001), and epochs (30 epochs).
- **Results:** The LSTM model with a configuration of `[300-128-128-6]` and dropout achieved the best performance:
  - **Precision:** 82.52%
  - **F1-Score:** 82.59%

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/your-repository.git
```

Navigate to the project directory:

```bash
cd your-repository
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
Note: The requirements.txt file should list necessary libraries such as PyTorch, NumPy, and any other dependencies.

## Usage
To run the project, use the provided Jupyter notebook:

Launch the Jupyter Notebook:

```bash
jupyter notebook projectNLP.ipynb
```
This notebook contains all the steps for data preprocessing, model building, training, and evaluation.

Alternatively, run experiments via the command line:

```bash
python train.py --model lstm --epochs 30 --lr 0.001
```
(Ensure your command matches the structure and arguments defined in your project scripts.)

## Future Work
Future improvements include:

Integrating advanced NLP models such as BERT.
Enhancing embedding techniques for richer feature representations.
Expanding the dataset and exploring additional data augmentation techniques.
Further hyperparameter tuning and model comparison.

## Authors
Hassan KHADRA - UA
Elias-Charbel SALAMEH - USEK

## License
This project is licensed under the MIT License. See the LICENSE file for details.