import sqlite3
conn=sqlite3.connect("sandbox.db")
cursor=conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS documents(
id INTEGER PRIMARY KEY,
content TEXT, 
embedding TEXT
)""")
cursor.execute("INSERT INTO documents(content,embedding) VALUES(?,?)",("Foundry Local runs AI models on your device.", "test1"))
cursor.execute("INSERT INTO documents(content,embedding) VALUES(?,?)",("SQLite is a serverless local database.", "test2"))
cursor.execute("INSERT INTO documents(content,embedding) VALUES(?,?)",("RAG combines retrieval with generation.", "test3"))
cursor.execute("INSERT INTO documents(content,embedding) VALUES(?,?)",("RAG is a serverless local database.", "test3"))
conn.commit()

cursor.execute("SELECT content FROM documents WHERE content LIKE ?", ("%RAG%",))
sonuc= cursor.fetchall()
print(sonuc)