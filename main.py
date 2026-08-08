from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    # Konfigürasyon oluştur
    config = Configuration(app_name="rag_project")
    manager = FoundryLocalManager(config)
    
    # Modeli indir ve başlat
    model_id = manager.download_model("phi-3.5-mini")
    
    # Modele soru sor
    client = manager.get_openai_client()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Merhaba! Kendini tanıtır mısın?"}]
    )
    
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()