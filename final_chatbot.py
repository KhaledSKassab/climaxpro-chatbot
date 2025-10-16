import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy.sparse import hstack, csr_matrix
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import joblib, json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from typing import TypedDict, Optional
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib, json, logging, traceback
app = Flask(__name__)
CORS(app)

LOG_FILE = "app_error.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
     return jsonify(status="ok")

AZURE_OPENAI_API_KEY = 'a920046601144cfeb17de716f9dc8610'
AZURE_OPENAI_ENDPOINT = "https://mopchatbot.openai.azure.com/"
AZURE_OPENAI_MODEL_NAME = "gpt-4.1-mini"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
# https://mopchatbot.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview

# Key: a920046601144cfeb17de716f9dc8610
llm = AzureChatOpenAI(
    model_name=AZURE_OPENAI_MODEL_NAME,
    temperature=0,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


def get_embedding(text):
    return embedding.embed_query(text)

with open("festival_info.json", encoding="utf-8") as f:
    festivals = json.loads(f.read())

df = pd.DataFrame(festivals)
original_df = pd.read_csv("List of film festivals.csv")
festival_embeddings = np.load("festival_embeddings.npy")
embedding_dim = len(festival_embeddings[0])

index = faiss.IndexFlatL2(embedding_dim)
index.add(festival_embeddings)

def recommend_festivals(description, top_k=3, weight_semantic=0.7):
    # Embed film description
    query_embedding = np.array(get_embedding(description)).reshape(1, -1)
    
    # Semantic similarity
    similarities = cosine_similarity(query_embedding, festival_embeddings)[0]
    
    # Keyword score: overlap of words between film and festival themes/genres
    description_words = set(description.lower().split())
    keyword_scores = []
    for _, row in df.iterrows():
        keywords = set(" ".join(row["themes_focus"] + row["genres_focus"]).lower().split())
        overlap = len(description_words & keywords)
        keyword_scores.append(overlap)
    keyword_scores = np.array(keyword_scores)
    keyword_scores = keyword_scores / (keyword_scores.max() or 1)  # normalize
    
    # Weighted combination
    combined_score = weight_semantic * similarities + (1 - weight_semantic) * keyword_scores
    
    # Top results
    top_indices = combined_score.argsort()[::-1][:top_k]
    return df.iloc[top_indices][["name", "festival_type", "themes_focus", "genres_focus"]]


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    film_description = data.get('query', '')
    if not film_description:
        return jsonify({"error": "query is required"}), 400
    
    recommendations = recommend_festivals(film_description, top_k=5)
    
    if len(recommendations[recommendations["name"] == "El Gouna Film Festival"]) == 0:
        recommendations = pd.concat([recommendations, df[df["name"] == "El Gouna Film Festival"][["name", "festival_type", "themes_focus", "genres_focus"]]])
    festinal_names = recommendations["name"].to_list()

    detailed_info = original_df[original_df["Name"].isin(festinal_names)]
    detailed_info = detailed_info.drop(columns=["Fest START Date", "Fest END Date", "Deadline Regular", "Deadline Extended"])

    all_info = pd.merge(recommendations, detailed_info, left_on="name", right_on="Name", how="left")
    all_info = all_info.drop(['Name'], axis=1)

    prompt = [
        {"role": "system", "content": """You are a helpful assistant.
        user descripe a film you will answer with recommendation film festival that can be submitted to.
        you will be given a festivals in context with details and convert it to human-language as a storyteller for the user in funny way and with cinematic and funny emojis.
        you will give the user all details about the festivals.
        give the user all festivals that in the context.
        ! IMPORTANT RULE... if film desciption in arabic answer in arabic language, if film desciption in english answer in english language, etc. even if the festival details are in different languages."""},
        {"role": "user", "content": "Festival data context: " + all_info.to_string(index=False) + "\n\nfilm description: " + film_description }
    ]

    res = llm.stream(prompt)

    def generate(respose_stream):
        answer = ""
        for i in respose_stream:
            print(i)
            answer += i.content
            yield i.content
        print(answer)
    return Response(generate(res), content_type='text/event-stream')



if __name__ == '__main__':
    app.run(port=5050, debug=True)