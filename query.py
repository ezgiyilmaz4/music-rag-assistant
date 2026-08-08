import json
import sqlite3
import math
from foundry_local_sdk import Configuration,FoundryLocalManager

#initializes the Foundry Local service
config=Configuration(app_name="foundry_local_rag")
FoundryLocalManager.initialize(config)
manager=FoundryLocalManager.instance

#converts text into numerical vectors
embedding_model=manager.catalog.get_model("qwen3-embedding-0.6b")
embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
embedding_model.load()
embedding_client=embedding_model.get_embedding_client()

# generates the final answer based on the retrieved context
chat_model=manager.catalog.get_model("phi-3.5-mini")
chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True))
chat_model.load()
chat_model=chat_model.get_chat_client()

# load documents that were already embedded and stored by ingest.py
conn=sqlite3.connect("ingest.db")
cursor=conn.cursor()
cursor.execute("SELECT title,text,embedding FROM documents")
docs=cursor.fetchall()

# dot product of two vectors
def dot_product(a,b):
    total = 0
    for i in range(len(a)):
        total += a[i]*b[i]
    return total

# magnitude(length) of a vector
def magnitude(a):
    total = 0
    for i in range(len(a)):
        total += a[i]**2
    total = math.sqrt(total)
    return total  

# measures the similarity between two embedding vectors
def cosine_similarity(A,B):
    cs=dot_product(A,B) / (magnitude(A) * magnitude(B))
    return cs 

# finds the top k most similar documents to the user's question
def find_relevant(question_embedding,docs,k):
    results=[]
    for i in range(len(docs)):
        copy=json.loads(docs[i][2])
        score = cosine_similarity(question_embedding,copy)
        results.append((score,docs[i][0],docs[i][1]))
    results.sort(reverse=True)
    return results[0:k]

# combines the retrieved chunks into a single context block
def generate_answers(question,relevant_chunks):
    context="\n\n".join(relevant_chunks)
    messages=[
        {"role":"system","content":"You are an assistant who responds based only on the information provided. Do not answer the question if you don't have the corresponding data in the context. If the answer is not in the context, say that you don't have enough information to answer."},
        {"role": "user" , "content":f"Context:\n{context}\n\nSoru: {question}"}
    ]
    response=chat_model.complete_chat(messages)
    return response.choices[0].message.content

# runs the full RAG pipeline    
def answer_query(user_question):
    response=embedding_client.generate_embeddings([user_question.lower()])
    c=find_relevant(response.data[0].embedding,docs,3)
    if(c[0][0]<0.40):  # if even the best match is below the threshold , we reject it without calling the chat model
        return "I do not know the answer." , c
    c1=[]
    for i in range(len(c)):
        if(c[i][0]>=0.40):  # it's not enough for top result to pass the threshold,every chunk that goes into the context must relevant on its own
            c1.append(c[i][2])
    answer=generate_answers(user_question,c1)
    return answer,c 
if __name__=="__main__":
    while True:
        user_question=input("Ask a question. Type 'exit' to exit: ")

        if user_question.lower()=="exit":
            break

        answer,chunks=answer_query(user_question)

        print("\nRetrieved sources:")
        for score,title,text in chunks:
            print(f"- {title}: {score:.4f}")

        print(f"\nAnswer:\n{answer}\n")