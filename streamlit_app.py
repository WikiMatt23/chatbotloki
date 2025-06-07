import streamlit as st
from openai import OpenAI, OpenAIError
import mimetypes

# --- TITLE AND INSTRUCTIONS ---
st.title("💬 Smart Multi-Model Chatbot")
st.write(
    "Choose a GPT model (or type a custom one) and upload documents/images for analysis. "
    "You need an [OpenAI API key](https://platform.openai.com/account/api-keys)."
)

# --- API KEY INPUT ---
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# --- CREATE CLIENT ---
try:
    client = OpenAI(api_key=openai_api_key)
    models_response = client.models.list()
    available_models = sorted([m.id for m in models_response.data if "gpt" in m.id])
except OpenAIError as e:
    st.error(f"❌ Failed to fetch models: {str(e)}")
    st.stop()

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

# --- MODEL SELECTION ---
model_labels = [cost_label(m) for m in available_models]
selected_label = st.selectbox("🧠 Choose a model", model_labels)
selected_model = selected_label.split(" ")[0]  # Extract just the model ID

# --- CUSTOM MODEL INPUT ---
custom_model_input = st.text_input("Or type a custom model name (overrides dropdown)", placeholder="e.g. gpt-4-vision-preview")

# Use custom model if provided
model_name = custom_model_input.strip() if custom_model_input else selected_model

# --- FILE UPLOAD ---
uploaded_files = st.file_uploader(
    "📁 Upload files (images, PDFs, text)", accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "txt", "pdf"]
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Type your message..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # File context summary
    file_descriptions = []
    for file in uploaded_files:
        file_type, _ = mimetypes.guess_type(file.name)
        if file_type and file_type.startswith("image/"):
            file_descriptions.append(f"📷 Image uploaded: {file.name}")
        elif file_type == "application/pdf":
            file_descriptions.append(f"📄 PDF uploaded: {file.name}")
        elif file_type.startswith("text/"):
            text = file.read().decode("utf-8")
            file_descriptions.append(f"📄 Text file ({file.name}) preview:\n{text[:1000]}...")
        file.seek(0)  # Reset pointer

    if file_descriptions:
        prompt += "\n\nAttached file details:\n" + "\n".join(file_descriptions)

    # --- TRY CALLING OPENAI COMPLETION ---
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except OpenAIError as e:
        st.error(f"❌ API error: `{str(e)}`")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: `{str(e)}`")

