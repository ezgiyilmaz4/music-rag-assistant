import streamlit as st


# query.py içindeki RAG fonksiyonunu yalnızca bir kez yükler.
@st.cache_resource
def load_rag():
    from query import answer_query
    return answer_query


st.set_page_config(
    page_title="Music RAG Assistant",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Music RAG Assistant")
st.write("MusicNet verileri hakkında soru sorabilirsin.")

show_sources = st.sidebar.checkbox(
    "Kaynakları göster",
    value=True
)

# Sohbet geçmişi ilk açılışta oluşturulur.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Önceki mesajları ekranda göster.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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


# Kullanıcının yeni sorusunu al.
question = st.chat_input("Müzik hakkında bir soru sor...")

if question:
    # Kullanıcı mesajını kaydet ve göster.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    # RAG yanıtını üret.
    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):
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

                # Yanıtı ve kaynakları sonraki ekran yenilemelerinde
                # gösterebilmek için kaydet.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as error:
                st.error(f"Bir hata oluştu: {error}")