import json
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager
import pandas as pd

# load the MusicNet metadata
df = pd.read_csv("musicnet_metadata.csv")

documents = []
grouped = df.groupby(["composer", "catalog_name"])

for (composer, catalog_name), group in grouped:
    # extract shared inormation for the current composition
    composition = group["composition"].iloc[0] 
    ensemble = group["ensemble"].iloc[0]
    # collect all movement names belonging to same composition
    movements = group["movement"].tolist()
    movements_text = ", ".join(movements)
    # create a text that will late be embedded
    text = f"{composer}'s {composition} ({catalog_name}) is written for {ensemble}. It has {len(movements)} movements: {movements_text}."
    # create a readable title for the document
    title = f"{composer} - {composition}"
    # store the title and content together
    documents.append({"title": title, "content": text})
print(len(documents)) # display the number of generated documents
print(documents[0]["title"]) # display the title of the first document

texts=[]

# extract only the document contents for embedding generation
for i in range(len(documents)):
    texts.append(documents[i]["content"])

print(len(texts))
print(texts[0])

# initialize the Foundry Local SDK
config=Configuration(app_name="foundry_local_rag")
FoundryLocalManager.initialize(config)
manager=FoundryLocalManager.instance

# load the embedding model from the local catalog
embedding_model=manager.catalog.get_model("qwen3-embedding-0.6b")
embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
embedding_model.load()
embedding_client=embedding_model.get_embedding_client()

doc_embeddings=[]

# generate embeddings in batches to improve efficiency
for i in range(0,len(texts),10):
    batch=texts[i:i+10]
    for j in range(len(batch)):
        batch[j]=batch[j].lower()
    # generate embeddings for the current batch
    response=embedding_client.generate_embeddings(batch)

    # extract embedding vectors from the response
    batch_embeddings=[item.embedding for item in response.data]

    # add the generated vectors to the complete embedding list
    doc_embeddings.extend(batch_embeddings)

print(len(doc_embeddings))
print(len(doc_embeddings[0]))

# connect to the SQLite database
conn=sqlite3.connect("ingest.db")
cursor=conn.cursor()

# create the documents table 
cursor.execute("""CREATE TABLE IF NOT EXISTS documents(
id INTEGER PRIMARY KEY,
text TEXT,
title TEXT,
embedding TEXT
)""")

# remove any existing records before inserting new data
cursor.execute("DELETE FROM documents")

# store each document together with its embedding
for i in range(len(documents)):
    # convert the embedding vector to JSON format for storage
    emb_json=json.dumps(doc_embeddings[i])
    cursor.execute("INSERT INTO documents(text,title,embedding) VALUES(?,?,?)",(documents[i]["content"],documents[i]["title"],emb_json))

# save all changes to the database
conn.commit()

# verify how many records were inserted
cursor.execute("SELECT COUNT(*) FROM documents")
sonuc=cursor.fetchone()
print(sonuc)

# close the database connection
conn.close()