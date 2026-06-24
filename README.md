# Statistical Methods for Machine Learning - Experimental Project (A.Y. 2025/26)
**Instructor:** Nicolò Cesa-Bianchi  
**University of Milan (Unimi)** - Data Science in Economics

## Assignment 2: Calibration of Probabilities

### Project Overview
This project focuses on evaluating and improving the probability calibration of two standard machine learning models implemented **completely from scratch**:
1. **Logistic Regression** (optimized via Gradient Descent)
2. **Random Forest** (implemented via Bootstrapped Decision Trees)

The models are tested on two real-world financial risk and credit scoring datasets:
* **Dataset 1:** German Credit Risk Dataset (UCI)
* **Dataset 2:** Default of Credit Card Clients Dataset (UCI)

To improve the confidence estimates (probabilities) of the base models, two post-processing calibration techniques were developed from scratch using `numpy`:
* **Platt Scaling** (Sigmoid-based calibration)
* **Isotonic Regression** (implemented via the Pool Adjacent Violators Algorithm - PAVA)

### Repository Structure
* `Probability_Calibration_German_Credit.ipynb`: Complete Jupyter Notebook containing data preprocessing, from-scratch implementations, training, calibration loops, and evaluations.
* `german.csv.csv`: Raw data for the German Credit Risk dataset.
* `default_of_credit_card_clients.csv`: Raw data for the Credit Card Default dataset.
* `calibration_curves.png`: Reliability diagrams for Dataset 1.
* `calibration_curves_dataset2.png`: Reliability diagrams for Dataset 2.
* `Report.pdf`: The comprehensive LaTeX-compiled project report (to be added).

### Evaluation Metrics Implemented
* Classification Accuracy
* Log-Loss
* Brier Score

### How to Run
1. Ensure you have `numpy`, `pandas`, and `matplotlib` installed.
2. Clone this repository or download the files.
3. Run the cells sequentially in `Probability_Calibration_German_Credit.ipynb`.
