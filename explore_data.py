import pandas as pd

df = pd.read_csv("musicnet_metadata.csv")

documents = []
grouped = df.groupby(["composer", "catalog_name"])

for (composer, catalog_name), group in grouped:
    composition = group["composition"].iloc[0]
    ensemble = group["ensemble"].iloc[0]
    
    movements = group["movement"].tolist()
    movements_text = ", ".join(movements)
    
    text = f"{composer}'s {composition} ({catalog_name}) is written for {ensemble}. It has {len(movements)} movements: {movements_text}."
    title = f"{composer} - {composition}"
    
    documents.append({"title": title, "content": text})

print(len(documents))
print(documents[0])