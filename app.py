from flask import Flask, render_template, request
import os
import numpy as np
import openai
from openai.embeddings_utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

import gzip
COMPLETIONS_MODEL = "text-davinci-002"
openai.api_key = os.getenv('my_new_api_key_here')


with gzip.open('vectorized_data.npy.gz', 'rb') as f:
    data_array = np.load(f, allow_pickle=True)

def page_not_found(e):
  return render_template('404.html'), 404


app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
    
  if request.method == 'POST':
    prompt = request.form['prompt']
    question = prompt
    question_vector = get_embedding(question, engine='text-embedding-ada-002')
    question_vector_transform = np.array(question_vector).reshape(1, -1)
    similarity = []

    for each in data_array[0]:
        v2 = np.array(each).reshape(1, -1)
        # compute the cosine similarity
        cosinesimilarity = cosine_similarity(question_vector_transform, v2)[0][0]
        similarity.append(cosinesimilarity)

    similarity = np.array(similarity)
    
    # Get the sorted indices of y in descending order
    sorted_indices = np.argsort(-similarity)

    # Sort x and y based on sorted_indices
    information_this = ' '.join(data_array[1][sorted_indices[0:2]])
    # cosinesimilarity_this = similarity[sorted_indices[0]]

    prompt = f""" Answer the following question using only the context below. Answer it in professional way, if you are not able to answer the question from the below context try to use your own knowledge

    Context:
    {information_this}

    Q: {question}
    A:"""

    answer = openai.Completion.create(
        prompt=prompt,
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )["choices"][0]["text"].strip(" \n")

    return answer

  return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run()
