# Sarcasm Headline Classification — SimpleRNN (Keras)

Classifying news headlines as sarcastic or not using a SimpleRNN built in Keras, on the Sarcasm Headlines dataset (28,619 headlines, ~52%/48% class balance).

## What's in this notebook

- Kept just the `headline` (text) and `is_sarcastic` (label) columns
- Tokenized text with Keras Tokenizer (top 5,000 words)
- Padded/truncated every sequence to exactly 50 tokens
- Built the required architecture: Embedding(5000, 32) → SimpleRNN(32) → Dense(1, sigmoid)
- Compiled with Adam optimizer and binary_crossentropy loss
- Trained for 5 epochs with a validation split
- Plotted training vs. validation accuracy and loss curves
- Explained the vanishing gradient problem in plain RNNs

## Result

Test accuracy: **82.6%**. Training accuracy climbed to ~99% while validation accuracy peaked early (~84%) then declined to ~82.5%, a clear sign of overfitting, the model began memorizing training headlines rather than generalizing. Test accuracy lines up with the validation plateau, confirming that as the more honest performance measure.

## Files

- `RNN_Sarcasm_Classifier.ipynb` — the notebook
- `Sarcasm_Headlines_Dataset_v2.json` — the dataset
