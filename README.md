# 📱 Neural Network SMS Text Classifier

A deep learning model that classifies SMS messages as **"ham"** (legitimate) or **"spam"** (unsolicited), built with TensorFlow/Keras. Completed as part of the [freeCodeCamp Machine Learning with Python Certification](https://www.freecodecamp.org/learn/machine-learning-with-python/).

---

## Overview

This project trains a neural network on a labeled SMS dataset to perform binary text classification. Given any SMS message string, the model returns the predicted label (`"ham"` or `"spam"`) along with a confidence probability.

---

## Dataset

The project uses the **SMS Spam Collection dataset**, provided via the freeCodeCamp CDN as two TSV files:

- `train-data.tsv` – labeled training messages
- `valid-data.tsv` – labeled validation/test messages

Each file contains two columns: the label (`ham` or `spam`) and the raw message text. The dataset is imbalanced, with ham messages significantly outnumbering spam.

---

## How It Works

1. **Data Loading** – Downloads `train-data.tsv` and `valid-data.tsv` from the freeCodeCamp CDN.
2. **Preprocessing** – Labels are encoded as binary values (`ham=0`, `spam=1`). Text is tokenized and padded using `TextVectorization`.
3. **Model Architecture** – A sequential Keras model consisting of:
   - `Embedding` layer (vocabulary size: 10,000, dimension: 128)
   - `GlobalAveragePooling1D` to produce a fixed-length vector
   - `Dense` layer (64 units, ReLU activation) with 50% dropout
   - `Output` layer (1 unit, sigmoid activation) for spam probability
4. **Training** – Compiled with `Adam` optimizer and `binary_crossentropy` loss, trained for ~10 epochs. Typically achieves ~98% validation accuracy.
5. **Prediction Function** – `predict_message(text)` returns a list `[probability, label]`.

---

## Usage

Open the notebook in Google Colab or Jupyter:

```bash
fcc_sms_text_classification.ipynb
```

Call the prediction function:

```python
predict_message("sale today! to stop texts call 98912460324")
# → [0.97, 'spam']

predict_message("how are you doing today")
# → [0.08, 'ham']
```

The function returns:
- **Element 0** – A float between 0 and 1 (closer to 1 = more likely spam)
- **Element 1** – The string `"ham"` or `"spam"`

---

## Test Cases

The notebook includes a built-in test suite that validates the model against 7 messages:

| Message | Expected |
|---|---|
| "how are you doing today" | ham |
| "sale today! to stop texts call 98912460324" | spam |
| "you have won £1000 cash! call to claim your prize." | spam |
| "i'll bring it tomorrow. don't forget the milk." | ham |
| "our new mobile video service is live..." | spam |

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- pandas
- numpy

Install dependencies:

```bash
pip install tensorflow pandas numpy
```

> GPU support is recommended for faster training. The notebook is designed to run end-to-end in Google Colab.

---

## Project Structure

```
fcc_sms_text_classification/
├── fcc_sms_text_classification.ipynb   # Main notebook
└── LICENSE                             # MIT License
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- [freeCodeCamp](https://www.freecodecamp.org/) for the project challenge and starter code
- UCI Machine Learning Repository for the original SMS Spam Collection dataset
