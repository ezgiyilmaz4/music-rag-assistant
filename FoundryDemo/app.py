from foundry_local_sdk import Configuration, FoundryLocalManager

FoundryLocalManager.initialize(
    Configuration(app_name="hello_foundry")
)

manager = FoundryLocalManager.instance

model = manager.catalog.get_model("qwen2.5-0.5b")

model.download()

model.load()

client = model.get_chat_client()

messages = [
    {
        "role": "user",
        "content": "Hello, world"
    }
]

for chunk in client.complete_streaming_chat(messages):
    print(chunk.choices[0].delta.content or "", end="")

model.unload()