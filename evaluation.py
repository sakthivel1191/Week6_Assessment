import os
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from api import app
load_dotenv()
client = TestClient(app)
headers = {
    "api-key": os.getenv("API_KEY")
}
document_folder = "documents"
for filename in os.listdir(document_folder):
    path = os.path.join(document_folder, filename)
    if filename.endswith((".txt", ".pdf", ".docx")):
        with open(path, "rb") as file:
            response = client.post(
                "/documents",
                headers=headers,
                files={
                    "file": (
                        filename,
                        file.read()
                    )
                }
            )
            print(response.json())
questions = [
    {"query": "Employee company rules", "source": "policy.txt", "page": 1},
    {"query": "Employee attendance", "source": "employee.txt", "page": 1},
    {"query": "Office safety rules", "source": "safety.txt", "page": 1},
    {"query": "Leave application process", "source": "employee.txt", "page": 1},
    {"query": "Company working hours", "source": "policy.txt", "page": 1},
    {"query": "Fire emergency procedure", "source": "safety.txt", "page": 1},
    {"query": "Employee responsibilities", "source": "rules.txt", "page": 1},
    {"query": "Workplace behavior", "source": "policy.txt", "page": 1},
    {"query": "Office equipment usage", "source": "manual.txt", "page": 1},
    {"query": "Employee benefits", "source": "employee.txt", "page": 1},
    {"query": "First aid procedure", "source": "safety.txt", "page": 1},
    {"query": "Computer usage instructions", "source": "manual.txt", "page": 1},
    {"query": "Company discipline rules", "source": "rules.txt", "page": 1},
    {"query": "Holiday policy", "source": "policy.txt", "page": 1},
    {"query": "Workplace security", "source": "safety.txt", "page": 1}
]
def evaluate(top_k):
    hits = 0
    reciprocal_rank = 0
    for question in questions:
        response = client.post(
            "/search",
            headers=headers,
            json={
                "query": question["query"],
                "top_k": top_k
            }
        )
        results = response.json()["results"]
        for rank, result in enumerate(results, start=1):
            if (
                result["source"] == question["source"]
                and result["page"] == question["page"]
            ):
                hits += 1
                reciprocal_rank += 1 / rank
                break
    hit_rate = hits / len(questions)
    mrr = reciprocal_rank / len(questions)
    return hit_rate, mrr
print("\nRETRIEVAL EVALUATION")
for k in [1, 3, 5]:
    hit_rate, mrr = evaluate(k)
    print("\nHit@", k, ":", round(hit_rate, 2))
    print("MRR:", round(mrr, 2))
