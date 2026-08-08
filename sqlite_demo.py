import sqlite3

conn= sqlite3.connect("test.db")
cursor=conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS belgeler(id INTEGER, icerik TEXT)")

cursor.execute("INSERT INTO belgeler VALUES(?,?)",(1,"kedicik sut seviyor"))
cursor.execute("INSERT INTO belgeler VALUES(?,?)",(2,"kedi sut iciyor"))
cursor.execute("INSERT INTO belgeler VALUES(?,?)",(3,"borsa cok artti"))
conn.commit()
sonuclar = cursor.execute("SELECT * FROM belgeler").fetchall()
print(sonuclar)
