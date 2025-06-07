import streamlit as st
from openai import OpenAI, OpenAIError
import mimetypes
import uuid

st.set_page_config(page_title="💬 Smart Multi-Model Chatbot", layout="wide")

# --- TITLE AND INSTRUCTIONS ---
st.title("💬 Smart Multi-Model Chatbot")
st.write(
    "Choose a GPT model (or type a custom one) and upload documents/images for analysis. "
    "You need an [OpenAI API key](https://platform.openai.com/account/api-keys)."
)

# --- API KEY INPUT ---
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password", key="api_key")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# --- CREATE CLIENT AND FETCH MODELS ---
try:
    client = OpenAI(api_key=openai_api_key)
    models_response = client.models.list()
    available_models = sorted([m.id for m in models_response.data if "gpt" in m.id])
except OpenAIError as e:
    st.error(f"❌ Failed to fetch models: {str(e)}")
    st.stop()

# --- SET DEFAULT MODEL ---
default_model = "gpt-4.1-mini"
if default_model in available_models:
    selected_model = default_model
else:
    selected_model = available_models[0]

# --- MODEL COST DISPLAY ---
def cost_label(model_name):
    if "3.5" in model_name:
        return f"{model_name} ($)"
    elif "gpt-4-turbo" in model_name:
        return f"{model_name} ($$)"
    elif "gpt-4" in model_name:
        return f"{model_name} ($$$)"
    else:
        return f"{model_name} (?)"

# --- SESSION STATE SETUP ---
if "chat_sessions" not in st.session_state:
    # Dict of chat_id -> messages list
    st.session_state.chat_sessions = {}

if "current_chat_id" not in st.session_state:
    # Initialize with a default chat
    default_id = str(uuid.uuid4())
    st.session_state.chat_sessions[default_id] = []
    st.session_state.current_chat_id = default_id

# Helper functions
def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = []
    st.session_state.current_chat_id = chat_id

def delete_current_chat():
    cid = st.session_state.current_chat_id
    if cid in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[cid]
    # Pick another chat if any remain, else create new
    if st.session_state.chat_sessions:
        st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions.keys()))
    else:
        new_chat()

# --- SIDEBAR: CHAT SESSION MANAGEMENT ---
with st.sidebar:
    st.header("💾 Chat Sessions")

    chat_ids = list(st.session_state.chat_sessions.keys())
    chat_names = [f"Chat {i+1}" for i in range(len(chat_ids))]

    # Select current chat
    if chat_ids:
        selected_index = chat_ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_ids else 0
        selected = st.selectbox("Select Chat", chat_names, index=selected_index)
        selected_chat_id = chat_ids[chat_names.index(selected)]
        if selected_chat_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_chat_id
    else:
        st.write("No chats available.")

    # Buttons for new and delete
    if st.button("➕ New Chat"):
        new_chat()
    if st.button("🗑️ Delete Current Chat"):
        delete_current_chat()

    st.markdown("---")

    # MODEL SELECTION IN SIDEBAR
    model_labels = [cost_label(m) for m in available_models]
    default_index = next((i for i, m in enumerate(available_models) if m == selected_model), 0)
    selected_label = st.selectbox("🧠 Choose a model", model_labels, index=default_index)
    selected_model_sid = selected_label.split(" ")[0]

    custom_model_input = st.text_input(
        "Or type a custom model name (overrides dropdown)",
        placeholder="e.g. gpt-4-vision-preview",
        key="custom_model"
    )

# Use model chosen in sidebar (custom overrides dropdown)
model_name = custom_model_input.strip() if custom_model_input else selected_model_sid

# --- MAIN PAGE ---

# Upload files (outside chat loop so they don't reset each input)
uploaded_files = st.file_uploader(
    "📁 Upload files (images, PDFs, text) for context",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "txt", "pdf"],
    key=st.session_state.current_chat_id + "_files"
)

# Load current chat messages
messages = st.session_state.chat_sessions.get(st.session_state.current_chat_id, [])

# Display chat history for current chat
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Type your message..."):
    messages.append({"role": "user", "content": prompt})
    st.session_state.chat_sessions[st.session_state.current_chat_id] = messages

    with st.chat_message("user"):
        st.markdown(prompt)

    # Add file context descriptions to prompt if files uploaded
    file_descriptions = []
    if uploaded_files:
        for file in uploaded_files:
            file_type, _ = mimetypes.guess_type(file.name)
            if file_type and file_type.startswith("image/"):
                file_descriptions.append(f"📷 Image uploaded: {file.name}")
            elif file_type == "application/pdf":
                file_descriptions.append(f"📄 PDF uploaded: {file.name}")
            elif file_type and file_type.startswith("text/"):
                text = file.read().decode("utf-8")
                preview = text[:1000].replace("\n", " ") + ("..." if len(text) > 1000 else "")
                file_descriptions.append(f"📄 Text file ({file.name}) preview:\n{preview}")
            file.seek(0)  # Reset file pointer for reuse

    if file_descriptions:
        full_prompt = prompt + "\n\nAttached file details:\n" + "\n".join(file_descriptions)
    else:
        full_prompt = prompt

    # Update last user message with full prompt (for context)
    messages[-1]["content"] = full_prompt
    st.session_state.chat_sessions[st.session_state.current_chat_id] = messages

    # Call OpenAI streaming completion
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )

        assistant_response = ""
        with st.chat_message("assistant"):
            for chunk in stream:
                content_chunk = chunk.choices[0].delta.get("content", "")
                assistant_response += content_chunk
                st.write(content_chunk, end="")

        messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.chat_sessions[st.session_state.current_chat_id] = messages

    except OpenAIError as e:
        st.error(f"❌ API error: `{str(e)}`")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: `{str(e)}`")
