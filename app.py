from flask import Flask, render_template, request
import config
import openai
import aiapi
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
import os


COMPLETIONS_MODEL = "text-davinci-003"
# EMBEDDINGS_MODEL = "text-embedding-ada-002"
openai.api_key = os.getenv('my_api_key')
import gzip

with gzip.open('vectorized_data.npy.gz', 'rb') as f:
    data_array = np.load(f, allow_pickle=True)

def page_not_found(e):
  return render_template('404.html'), 404


app = Flask(__name__)
app.config.from_object(config.config['development'])

app.register_error_handler(404, page_not_found)


@app.route('/', methods = ['POST', 'GET'])
def index():
    
    if request.method == 'POST':
      prompt = request.form['prompt']
      answer = aiapi.get_response_of_text(prompt)
      return answer

    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)
