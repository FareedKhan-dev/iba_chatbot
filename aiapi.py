from app import *

def get_response_of_text(prompt):
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

    prompt = f""" Answer only IBA related question using only the context below. If someone asked about your information reply to them exact text "I am an IBA CHATBOT, who solve your queries", Answer it in professional way, if you are not able to answer the question make sure you don't give any wrong information

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