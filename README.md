# ✈️ Travel Sentiment Insights — Airline Tweet Sentiment Analysis

A deep learning project that performs **binary sentiment classification** (positive / negative) on US airline tweets using an LSTM neural network.

---

## 📌 Overview

Customer feedback on social media provides valuable insights for the airline industry. This project uses a Long Short-Term Memory (LSTM) network to classify airline-related tweets as positive or negative, helping extract actionable sentiment insights from traveler reviews.

---

## 📊 Dataset

| File | Description | Size |
|---|---|---|
| `Tweets.csv` | US airline tweet reviews | ~3.4 MB |
| `database.sqlite` | SQLite database (source data) | ~5 MB |

The dataset contains tweet text along with sentiment labels (`positive`, `negative`, `neutral`). Neutral tweets are filtered out for binary classification.

---

## ⚙️ Methodology

1. **Data Filtering** — Remove neutral tweets for binary classification
2. **Tokenization** — Keras `Tokenizer` with vocabulary size of 5,000
3. **Padding** — Sequences padded/truncated to length 200
4. **Model Architecture**
   ```
   Embedding (32-dim) → SpatialDropout1D (0.25) → LSTM (50 units) → Dropout (0.2) → Dense (sigmoid)
   ```
5. **Training** — Binary cross-entropy loss, Adam optimizer, 5 epochs, batch size 32
6. **Validation** — 80/20 train/validation split

---

## 📈 Results

Training and validation curves are saved as:

- `Accuracy plot.jpg` — Training vs validation accuracy
- `Loss plot.jpg` — Training vs validation loss

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the script

```bash
python "Sentiment Analysis.py"
```

### Or use the notebook

Open `Sentiment Analysis.ipynb` in Jupyter Notebook or Google Colab for an interactive walkthrough with training visualizations.

### Predict on custom text

```python
predict_sentiment("I enjoyed my journey on this flight.")
# Output → Predicted label: positive

predict_sentiment("This is the worst flight experience of my life!")
# Output → Predicted label: negative
```

---

## 🐛 Bugs Fixed

The following bug was found and fixed in `Sentiment Analysis.py`:

| # | Line | Bug | Fix |
|---|---|---|---|
| 1 | 18 | Stray `ra` characters after `pd.read_csv("./Tweets.csv")ra` causing a `SyntaxError` | Removed the stray characters |

---

## 📁 Project Structure

```
Travel Sentiment Insights/
├── Sentiment Analysis.ipynb    # Interactive notebook
├── Sentiment Analysis.py       # Standalone Python script
├── Tweets.csv                  # Tweet dataset
├── database.sqlite             # SQLite source data
├── Accuracy plot.jpg           # Accuracy curve
├── Loss plot.jpg               # Loss curve
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📜 License

This project is for educational and research purposes.
