import numpy as np

# Basit "sahte" embedding fonksiyonu (gerçek modelin yerine, mantığı göstermek için)
# Gerçek hayatta bu işi Foundry Local'daki embedding modeli yapar
def basit_embedding(cumle):
    # Her kelimenin uzunluğuna göre basit bir vektör üretiyoruz (sadece örnek amaçlı!)
    kelimeler = cumle.lower().split()
    vektor = np.zeros(10)
    for kelime in kelimeler:
        for harf in kelime:
            vektor[ord(harf) % 10] += 1
    return vektor / (np.linalg.norm(vektor) + 1e-9)  # normalize et

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Test cümleleri
cumleler = [
    "Kedi süt içiyor",
    "Kedicik süt içmeyi seviyor",
    "Borsa bugün düştü"
]

# Her cümlenin embedding'ini hesapla
embeddingler = [basit_embedding(c) for c in cumleler]

# Sorgu
sorgu = "Kedi süt içmeyi seviyor"
sorgu_embedding = basit_embedding(sorgu)

print(f"Sorgu: '{sorgu}'\n")
print("Benzerlik skorları:")
for cumle, emb in zip(cumleler, embeddingler):
    benzerlik = cosine_similarity(sorgu_embedding, emb)
    print(f"  '{cumle}' -> {benzerlik:.3f}")