import sqlite3
conn=sqlite3.connect("ingest.db")
cursor=conn.cursor()
cursor.execute("SELECT id,title FROM documents")
conclusion=cursor.fetchall()
for satir in conclusion:
    print(conclusion)
conn.close()