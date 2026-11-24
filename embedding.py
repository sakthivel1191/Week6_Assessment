import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_similarity
from sklearn.cluster import KMeans
load_dotenv()
sentences = [
    "Python is a programming language",
    "Java is used for software development",
    "A doctor treats sick people",
    "A physician cares for ill patients",
    "The sun rises in the east",
    "The moon appears at night",
    "Machine learning is a part of AI",
    "Artificial intelligence helps computers learn",
    "Employees must follow company policies",
    "Workers should obey office rules"
]
model = SentenceTransformer("all-MiniLM-L6-v2")
start = time.time()
local_embeddings = model.encode(sentences)
local_time = time.time() - start
print("\nLOCAL EMBEDDING")
print("Dimensions:", local_embeddings.shape)
print("Time:", round(local_time, 4), "seconds")
api_key = os.getenv("OPENAI_API_KEY")
api_embeddings = None
if api_key:
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        start = time.time()
        response = client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=sentences
        )
        api_embeddings = np.array(
            [item.embedding for item in response.data]
        )
        api_time = time.time() - start
        print("\nOPENROUTER API EMBEDDING")
        print("Dimensions:", api_embeddings.shape)
        print("Time:", round(api_time, 4), "seconds")
    except Exception as error:
        print("\nAPI EMBEDDING ERROR")
        print(error)
        print("Continuing with local embeddings.")
else:
    print("\nAPI key not found. Skipping API embedding.")
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)
print("\nCOSINE SIMILARITY TESTS")
print("[1,2,3] vs [2,4,6] ->",cosine_similarity([1, 2, 3], [2, 4, 6]))
print("[1,0] vs [0,1] ->",cosine_similarity([1, 0], [0, 1]))
print("[1,0] vs [-1,0] ->",cosine_similarity([1, 0], [-1, 0]))
print("\nSCIKIT-LEARN COMPARISON")
print(sklearn_similarity([[1, 2, 3]],[[2, 4, 6]])[0][0])
similarity_matrix = sklearn_similarity(local_embeddings)
print("\nSIMILARITY MATRIX")
print(np.round(similarity_matrix, 2))
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = similarity_matrix[i][j]
        pairs.append(
            (sentences[i], sentences[j], score)
        )
pairs.sort(key=lambda x: x[2], reverse=True)
print("\nTOP 3 SIMILAR PAIRS")
for first, second, score in pairs[:3]:
    print("Sentence 1:", first)
    print("Sentence 2:", second)
    print("Similarity:", round(score, 3))
    print()
print("\nNO COMMON WORDS EXAMPLE")
print("Sentence 1:", sentences[2])
print("Sentence 2:", sentences[3])
print("Similarity:",round(cosine_similarity(local_embeddings[2],local_embeddings[3]),3))
print("\nSEMANTIC SEARCH")
queries = [
    "How should employees behave at work?",
    "What is artificial intelligence?",
    "Who helps people when they are ill?"
]
for query in queries:
    query_embedding = model.encode([query])
    scores = sklearn_similarity(
        query_embedding,
        local_embeddings
    )[0]
    best_index = np.argmax(scores)
    print("\nQuery:", query)
    print("Result:", sentences[best_index])
    print("Score:", round(scores[best_index], 3))
print("\nDUPLICATE DETECTION")
threshold = 0.75
for first, second, score in pairs:
    if score >= threshold:
        print(first)
        print(second)
        print("Similarity:", round(score, 3))
        print()
print("\nK-MEANS CLUSTERING")
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(local_embeddings)
for sentence, cluster in zip(sentences, clusters):
    print("Cluster", cluster, ":", sentence)
