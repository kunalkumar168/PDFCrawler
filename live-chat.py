import gradio as gr
import os
import json
from chatbot import ChatBot

# Define model names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral"

# Initialize chatbot
print("[INFO] Initializing ChatBot...")
chatbot = ChatBot(LLM_MODEL_NAME, EMBEDDING_MODEL_NAME)

# Load queries and expected answers
def load_query_dict(query_file_path):
    if not os.path.exists(query_file_path):
        print(f"[WARNING] Query file not found at {query_file_path}")
        return {}

    with open(query_file_path, "r") as file:
        data = json.load(file)
        return {
            item['question'].lower(): item['answer']
            for item in data.get('queries', [])
        }

current_workdir = os.getcwd()
query_dict = load_query_dict(os.path.join(current_workdir, "query.json"))

# Chat history storage
chat_history = []

def chat_with_bot(user_input):
    user_input_lower = user_input.lower()
    predicted_answer, sources = chatbot.response(user_input_lower, top_k=6)

    expected = query_dict.get(user_input_lower)
    validation_feedback = ""

    if expected:
        match_result = chatbot.validate_answers(predicted_answer, expected)
        validation_feedback = (
            f"\n\n📖 Expected: {expected}\n"
            f"🎯 Match: {'✅' if 'yes' in match_result.lower() else '❌'}"
        )

    sources_str = ", ".join(sources)
    response_text = f"{predicted_answer}{validation_feedback}\n\n📚 Sources: {sources_str}"

    chat_history.append((user_input, response_text))
    return "", chat_history

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Live Chatbot")

    chatbot_ui = gr.Chatbot(height=600)

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your question here...", 
            show_label=False
        )
        send_button = gr.Button("Send", variant="primary")

    send_button.click(fn=chat_with_bot, inputs=[msg_box], outputs=[msg_box, chatbot_ui])
    msg_box.submit(fn=chat_with_bot, inputs=[msg_box], outputs=[msg_box, chatbot_ui])

if __name__ == "__main__":
    demo.launch(share=True)  # Set to False for private interface
