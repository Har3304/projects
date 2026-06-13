# Inference Code
import os
import joblib
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

class WebClassifier:
  def __init__(self, model_dir: str = './saved_model'):
    if not os.path.exists(model_dir):
      raise FileNotFoundError(f'Model directory {model_dir} does not exist. Please check the model path')
    print('Loading assests....')
    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    self.model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
    self.classes = joblib.load(os.path.join(model_dir, 'encoder.pkl'))
  def predict(self, text: str):
    inputs = self.tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = self.model(inputs)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]

    predicted_class = np.argmax(probabilities)
    predicted_class_name = self.classes[predicted_class]
    cofidenece_score = float(probabilities[predicted_class])

    return {'predicted_class_name': predicted_class_name, 'cofidenece_score': cofidenece_score}
  

# if __name__=='__main__':
#   predictor = WebClassifier(model_dir='./saved_model')
#   sample_text = "Python programming tutorials, machine learning algorithms, and software development best practices."
  
#   result = predictor.predict(sample_text)
#   print(result)



