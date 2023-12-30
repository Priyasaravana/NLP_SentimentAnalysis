# NLP_SentimentAnalysis
Sentiment analysis for reddit comments
Emotional states including fear, rage, joy, and sadness can all have an impact on our mental well-being and ability to make decisions. This project will analyse the emotions in GoEmotions dataset presented in paper. About 58 thousand Reddit comments with human annotations assigned to 27 emotions or neutral are included in the original dataset. Our team has grouped the 27 emotions into the below 13 labels to better classify the customer review.
Joy
Amusement, excitement, joy
Anger
Anger, disapproval
Desire
Desire
Disgust
disgust, annoyance
Pride
Pride, Admiration, Relief
Sadness
Grief, remorse, sadness, embarrassment
Agreement
Approval, realization
Fear
Nervousness, Fear
Surprise
Surprise, curiosity
Optimism
Optimism, gratitude
Love
Love, caring
Disappointment
Disappointment
Confusion
Confusion
Neutral
Neutral

## Experiments

Conducted experiments with different model LSTM, RNN, Bi-LSTM and GRU

<!DOCTYPE html>
<html>
<head>
<style>
  table {
    font-family: Arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
  }

  th, td {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
  }

  th {
    background-color: #f2f2f2;
  }
</style>
</head>
<body>

<h2>Natural Language Processing Pipelines</h2>

<h3>Hypothesis</h3>
<p>I have constructed two pipelines to support my hypothesis. The pipelines are detailed below:</p>

<h4>Pipeline 1</h4>
<table>
  <tr>
    <th>S. No</th>
    <th>Model</th>
    <th>Stop words</th>
    <th>Tokenization</th>
    <th>Lemmatization</th>
    <th>Vocab counter</th>
    <th>Vectorization</th>
    <th>Split</th>
    <th>Architecture</th>
  </tr>
  <tr>
    <td>1</td>
    <td>LSTM</td>
    <td>Spacy</td>
    <td>Spacy Tokenizer</td>
    <td>Spacy Lemma attribute</td>
    <td>Count vectorizer (Sci-kit learn)</td>
    <td>TF-IDF (Sci-kit learn)</td>
    <td>80-10-10 (Manual split)</td>
    <td>TensorFlow Keras</td>
  </tr>
</table>

<h4>Pipeline 2</h4>
<table>
  <tr>
    <th>S. No</th>
    <th>Model</th>
    <th>Stop words</th>
    <th>Tokenization</th>
    <th>Lemmatization</th>
    <th>Vocab counter</th>
    <th>Vectorization</th>
    <th>Split</th>
    <th>Architecture</th>
  </tr>
  <tr>
    <td>2</td>
    <td>RNN</td>
    <td>NLTK</td>
    <td>NLTK - Word_tokenize</td>
    <td>WordNet Lemmatizer</td>
    <td>Frequency distribution</td>
    <td>Glove</td>
    <td>80-10-10 (Train-Test-Split sklearn)</td>
    <td>PyTorch BiLSTM GRU</td>
  </tr>
</table>

</body>
</html>
