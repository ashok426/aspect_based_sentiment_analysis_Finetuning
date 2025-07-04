# Fine-tuning BERT for Aspect-Based Sentiment Analysis (ABSA)
import pandas as pd
import tensorflow as tf
from transformers import (
  BertTokenizer,
  TFBertForSequenceClassification,
  create_optimizer
)
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Data Loading Functions
# =========================

def load_mams_data(path):
  df = pd.read_csv(path)
  df = df[['text', 'term', 'polarity']]
  return df

def load_laptop_data(path):
  text_ds, term_ds, polarity_ds = [], [], []
  with open(path, 'r') as f:
    data = f.read().splitlines()

  terms, polarities = [], []
  text, term, polarity = "", "", ""
  for row in data[2:]:
    if len(row) == 1:
      if term != "":
        terms.append(term)
        polarities.append(polarity)
      for i in range(len(terms)):
        text_ds.append(text.strip())
        term_ds.append(terms[i].strip())
        polarity_ds.append(polarities[i])
      text, term, polarity = "", "", ""
      terms, polarities = [], []
    else:
      token, label = row.rsplit(',', 1)
      text += " " + token
      if label == 'T-POS':
        label = 'positive'
      elif label == 'T-NEG':
        label = 'negative'
      elif label == 'T-NEU':
        label = 'neutral'
      else:
        if term != "":
          terms.append(term)
          polarities.append(polarity)
        term, polarity = "", ""
        continue
      if polarity == label:
        term += " " + token
      else:
        if term != "":
          terms.append(term)
          polarities.append(polarity)
        polarity = label
        term = token
  df = pd.DataFrame({'text': text_ds, 'term': term_ds, 'polarity': polarity_ds})
  return df

# =========================
# Data Preparation
# =========================

# Load datasets
laptop_train, laptop_test, laptop_valid = map(load_laptop_data, ['laptop_train.csv', 'laptop_test.csv', 'laptop_dev.csv'])
rest_train, rest_test, rest_valid = map(load_laptop_data, ['rest16_train.csv', 'rest16_test.csv', 'rest16_dev.csv'])
mams_train, mams_test, mams_valid = map(load_mams_data, ['mams_train.csv', 'mams_test.csv', 'mams_val.csv'])

# Combine and shuffle datasets
train_data = pd.concat([
  laptop_train, rest_train, mams_train,
  laptop_test, rest_test, mams_test
]).sample(frac=1).reset_index(drop=True)
valid_data = pd.concat([laptop_valid, rest_valid, mams_valid]).sample(frac=1).reset_index(drop=True)

# Remove neutral polarity
train_data = train_data[train_data['polarity'] != 'neutral']
valid_data = valid_data[valid_data['polarity'] != 'neutral']

# Encode labels and preprocess text
pol2idx = {'positive': 1, 'negative': 0}
for data in [train_data, valid_data]:
  data['polarity'] = data['polarity'].apply(lambda x: pol2idx[x])
  data['text'] = data['text'].apply(lambda x: x.lower())
  data['term'] = data['term'].apply(lambda x: str(x).lower()).astype('str')
  data.drop_duplicates(["text", "term", "polarity"], inplace=True)

# =========================
# Tokenization & Encoding
# =========================

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_sentence(sentence, aspect):
  text = f"[CLS] {aspect} [SEP] {sentence} [SEP]"
  return tokenizer.encode_plus(
    text,
    add_special_tokens=False,
    max_length=100,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='tf'
  )

def encode_examples(df):
  input_ids_list, attention_mask_list, label_list = [], [], []
  for _, row in df.iterrows():
    encoded = encode_sentence(row['text'], row['term'])
    input_ids_list.append(encoded['input_ids'])
    attention_mask_list.append(encoded['attention_mask'])
    label_list.append(row['polarity'])
  return (
    tf.concat(input_ids_list, axis=0),
    tf.concat(attention_mask_list, axis=0),
    tf.convert_to_tensor(label_list)
  )

train_input_ids, train_attention_masks, train_labels = encode_examples(train_data)
val_input_ids, val_attention_masks, val_labels = encode_examples(valid_data)

# =========================
# TensorFlow Dataset
# =========================

train_dataset = tf.data.Dataset.from_tensor_slices((
  {'input_ids': train_input_ids, 'attention_mask': train_attention_masks},
  train_labels
)).shuffle(1000).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
  {'input_ids': val_input_ids, 'attention_mask': val_attention_masks},
  val_labels
)).batch(16)

# =========================
# Model Setup & Training
# =========================

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_train_steps=5000, num_warmup_steps=500)

model.compile(
  optimizer=optimizer,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

model.fit(train_dataset, epochs=3, validation_data=val_dataset)

# =========================
# Save Model & Tokenizer
# =========================

model.save_pretrained('fine_tuned_bert_absa')
tokenizer.save_pretrained('fine_tuned_bert_absa')

# =========================
# Prediction Function
# =========================

loaded_tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert_absa')
loaded_model = TFBertForSequenceClassification.from_pretrained('fine_tuned_bert_absa')

def predict_sentiment(sentence, aspect):
  encoded = loaded_tokenizer.encode_plus(
    f"[CLS] {aspect.lower()} [SEP] {sentence.lower()} [SEP]",
    add_special_tokens=False,
    max_length=100,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='tf'
  )
  input_ids = encoded['input_ids']
  attention_mask = encoded['attention_mask']
  predictions = loaded_model({'input_ids': input_ids, 'attention_mask': attention_mask})[0]
  predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
  idx2pol = {0: 'negative', 1: 'positive'}
  return idx2pol[predicted_class]

# =========================
# Example Predictions
# =========================

sample_sentence1 = "The food was amazing, but the service was slow."
sample_aspect1 = "food"
sample_aspect2 = "service"
print(f"\nSentence: '{sample_sentence1}'")
print(f"Aspect: '{sample_aspect1}' -> Predicted Sentiment: {predict_sentiment(sample_sentence1, sample_aspect1)}")
print(f"Aspect: '{sample_aspect2}' -> Predicted Sentiment: {predict_sentiment(sample_sentence1, sample_aspect2)}")

sample_sentence2 = "The battery life is terrible on this laptop."
sample_aspect3 = "battery life"
print(f"\nSentence: '{sample_sentence2}'")
print(f"Aspect: '{sample_aspect3}' -> Predicted Sentiment: {predict_sentiment(sample_sentence2, sample_aspect3)}")


# =========================
# Evaluation Function
# =========================

def evaluate_absa(model, tokenizer, data_df):
  y_true = data_df['polarity'].tolist()
  y_pred = []
  for _, row in data_df.iterrows():
    pred = predict_sentiment(row['text'], row['term'])
    y_pred.append(1 if pred == 'positive' else 0)
  print("Accuracy:", accuracy_score(y_true, y_pred))
  print(classification_report(y_true, y_pred, target_names=['negative', 'positive']))

# Example usage:
# evaluate_absa(loaded_model, loaded_tokenizer, valid_data)

