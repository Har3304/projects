from flask import Flask, request
from inference import WebClassifier
import requests

app = Flask(__name__)

@app.route('/')
def home():
    html_form = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Web Classifier</title>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h2>Web Description Classifier</h2>
        
        <!-- The action matches our POST route, and the method is POST -->
        <form action="/category" method="POST">
            <label for="desc">Enter Web Description:</label><br><br>
            
            <!-- The name attribute MUST exactly match the key in request.form.get() -->
            <textarea id="desc" name="Web Description: " rows="6" cols="60" placeholder="Type or paste your text here..."></textarea><br><br>
            
            <button type="submit" style="padding: 10px 20px; font-size: 16px; cursor: pointer;">Classify Text</button>
        </form>
    </body>
    </html>
    """
    return html_form

@app.route('/category', methods=['POST'])
def identify():
    text = request.form.get('Web Description: ')
    predictor = WebClassifier()
    category = predictor.predict(text)
    return f"<h3>Prediction Result:</h3><p>{category}</p><br><a href='/'>Go Back</a>"

if __name__ == '__main__':
    app.run(debug=True)