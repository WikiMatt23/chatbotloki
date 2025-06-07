import streamlit as st
from openai import OpenAI, OpenAIError
import mimetypes
import tiktoken
import uuid

# --- MODEL PRICING AND TOKENIZATION ---
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015, "encoding": "cl100k_base"},
    "gpt-4": {"input": 0.03, "output": 0.06, "encoding": "cl100k_base"},
    "gpt-4-0613": {"input": 0.03, "output": 0.06, "encoding": "cl100k_base"},
    "gpt-4-32k": {"input": 0.06, "output": 0.12, "encoding": "cl100k_base"},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03, "encoding": "cl100k_base"},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03, "encoding": "cl100k_base"},
    "gpt-4-vision-preview": {"input": 0.01, "output": 0.03, "encoding": "cl100k_base"},
    "gpt-4o": {"input": 0.005, "output": 0.015, "encoding": "cl100k_base"},
}

def cost_label(model_name):
    if model_name in MODEL_PRICING:
        in_cost = MODEL_PRICING[model_name]["input"] * 1000000
        out_cost = MODEL_PRICING[model_name]["output"] * 1000000
        return f"{model_name} (${in_cost:.0f}/M in, ${out_cost:.0f}/M out)"
    return f"{model_name} (Pricing Unknown)"

def count_tokens(text, model_name):
    encoding_name = MODEL_PRICING.get(model_name, {}).get("encoding", "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

# --- TITLE AND INSTRUCTIONS ---
st.title("💬 Smart Multi-Model Chatbot")
st.write("Choose a GPT model and upload documents/images for analysis."
         " You need an [OpenAI API key](https://platform.openai.com/account/api-keys).")

# --- API KEY INPUT ---
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# --- OPENAI CLIENT ---
try:
    client = OpenAI(api_key=openai_api_key)
    models_response = client.models.list()
    available_models = sorted([m.id for m in models_response.data if "gpt" in m.id])
except OpenAIError as e:
    st.error(f"❌ Failed to fetch models: {str(e)}")
    st.stop()

# --- SESSION STATE INIT ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "chat_costs" not in st.session_state:
    st.session_state.chat_costs = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

# --- SIDEBAR CHAT SESSION NAV ---
st.sidebar.header("💾 Chat Sessions")

# Display all chats
for chat_id in list(st.session_state.chat_sessions):
    if st.sidebar.button(f"🗂️ Chat {chat_id[:5]}", key=chat_id):
        st.session_state.current_chat_id = chat_id

if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = []
    st.session_state.chat_costs[new_id] = 0.0
    st.session_state.current_chat_id = new_id

if st.sidebar.button("🗑️ Delete Current Chat"):
    cid = st.session_state.current_chat_id
    st.session_state.chat_sessions.pop(cid, None)
    st.session_state.chat_costs.pop(cid, None)
    st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions), str(uuid.uuid4()))

# Show total cost of current chat
cost = st.session_state.chat_costs.get(st.session_state.current_chat_id, 0.0)
st.sidebar.markdown(f"**💰 Estimated cost:** `${cost:.4f}`")

# --- MODEL SELECTION ---
def_model = "gpt-4o" if "gpt-4o" in available_models else available_models[0]
default_index = next((i for i, m in enumerate(available_models) if m == def_model), 0)
model_labels = [cost_label(m) for m in available_models]
selected_label = st.selectbox("🧠 Choose a model", model_labels, index=default_index)
selected_model = selected_label.split(" ")[0]
custom_model = st.text_input("Or type a custom model name", placeholder="e.g. gpt-4o")
model_name = custom_model.strip() if custom_model else selected_model

# --- FILE UPLOAD ---
uploaded_files = st.file_uploader("📁 Upload files (images, PDFs, text)",
    accept_multiple_files=True, type=["png", "jpg", "jpeg", "txt", "pdf"])

# --- DISPLAY CHAT HISTORY ---
chat_id = st.session_state.current_chat_id
messages = st.session_state.chat_sessions.setdefault(chat_id, [])

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Type your message..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    file_notes = []
    for f in uploaded_files:
        ftype, _ = mimetypes.guess_type(f.name)
        if ftype and ftype.startswith("image"):
            file_notes.append(f"📷 Image: {f.name}")
        elif ftype == "application/pdf":
            file_notes.append(f"📄 PDF: {f.name}")
        elif ftype and ftype.startswith("text"):
            txt = f.read().decode("utf-8")
            file_notes.append(f"📄 Text ({f.name}):\n{txt[:1000]}...")
        f.seek(0)

    if file_notes:
        prompt += "\n\nAttached file details:\n" + "\n".join(file_notes)
        messages[-1]["content"] = prompt

    # Estimate input tokens before sending
    input_tokens = sum(count_tokens(m["content"], model_name) for m in messages)
    input_price = MODEL_PRICING.get(model_name, {}).get("input", 0)
    output_price = MODEL_PRICING.get(model_name, {}).get("output", 0)

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )

        assistant_response = ""
        response_container = st.empty()

        with st.chat_message("assistant"):
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                assistant_response += delta
                response_container.markdown(assistant_response)

        messages.append({"role": "assistant", "content": assistant_response})

        # Estimate cost
        output_tokens = count_tokens(assistant_response, model_name)
        cost = input_tokens * input_price + output_tokens * output_price
        st.session_state.chat_costs[chat_id] = st.session_state.chat_costs.get(chat_id, 0.0) + cost

    except OpenAIError as e:
        st.error(f"❌ API error: `{str(e)}`")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: `{str(e)}`")
