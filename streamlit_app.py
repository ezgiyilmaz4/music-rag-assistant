import streamlit as st
import sqlite3,json,math
from foundry_local_sdk import Configuration,FoundryLocalManager


st.set_page_config(
    page_title="Music RAG assistant",
    page_icon="🎵",
    layout="centered"
)

def dot_product(a,b):
    total = 0
    for i in range(len(a)):
        total += a[i]*b[i]
    return total

def magnitude(a):
    total = 0
    for i in range(len(a)):
        total += a[i]**2
    total = math.sqrt(total)
    return total  

def cosine_similarity(A,B):
    cs=dot_product(A,B) / (magnitude(A) * magnitude(B))
    return cs 

def find_relevant(question_embedding,docs,k):
    results=[]
    for i in range(len(docs)):
        copy=json.loads(docs[i][2])
        score = cosine_similarity(question_embedding,copy)
        results.append((score,docs[i][0],docs[i][1]))
    results.sort(reverse=True)
    return results[0:k]

@st.cache_resource
def load_rag():
    with st.status("loading models...", expanded=True) as status:

        config=Configuration(app_name="foundry_local_rag")
        FoundryLocalManager.initialize(config)
        manager=FoundryLocalManager.instance

        def generate_answers(question,relevant_chunks):
         context="\n\n".join(relevant_chunks)
         messages=[
        {"role":"system","content":"You are an assistant who responds based only on the information provided. Do not answer the question if you don't have the corresponding data in the context. If the answer is not in the context, say that you don't have enough information to answer."},
        {"role": "user" , "content":f"Context:\n{context}\n\nQuestion: {question}"}
         ]
         response=chat_model.complete_chat(messages)
         return response.choices[0].message.content

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

        step1 = st.empty()
        embedding_model=manager.catalog.get_model("qwen3-embedding-0.6b")
        embedding_model.download(lambda p: step1.write(f"Downloading embedding model: {p:.1f}%"))
        embedding_model.load()
        embedding_client=embedding_model.get_embedding_client()

        step2 = st.empty()
        chat_model=manager.catalog.get_model("phi-3.5-mini")
        chat_model.download(lambda p: step2.write(f"Downloading chat model: {p:.1f}%"))
        chat_model.load()
        chat_model=chat_model.get_chat_client()

        step3 = st.empty()
        conn=sqlite3.connect("ingest.db")
        cursor=conn.cursor()
        cursor.execute("SELECT title,text,embedding FROM documents")
        docs=cursor.fetchall()
        step3.write(f"✅ {len(docs)} documents loaded")

    status.update(label="Ready!", state="complete")
    return answer_query
       

st.title("🎵 music RAG assistant")
st.write("you can ask questions about the MusicNet data.")

show_sources = st.sidebar.checkbox(
    "Show sources",
    value=True
)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    if message["role"] == "user":
        avatar="user_avatar.jpg"
    else:
        avatar="assistant_avatar.jpg"
    
    with st.chat_message(message["role"],avatar=avatar):
        st.write(message["content"])

        if (
            message["role"] == "assistant"
            and show_sources
            and message.get("sources")
        ):
            with st.expander("Retrieved sources"):
                for score, title, text in message["sources"]:
                    st.write(f"**{title}** — similarity: `{score:.4f}`")
                    st.caption(text)


question = st.chat_input("Ask a question about music...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user",avatar="user_avatar.jpg"):
        st.write(question)

    with st.chat_message("assistant",avatar="assistant_avatar.jpg"):
        with st.spinner("Preparing the answer..."):
            try:
                answer_query = load_rag()
                answer, sources = answer_query(question)

                st.write(answer)

                if show_sources:
                    with st.expander("Retrieved sources"):
                        for score, title, text in sources:
                            st.write(
                                f"**{title}** — similarity: `{score:.4f}`"
                            )
                            st.caption(text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as error:
                st.error(f"An error occurred: {error}")