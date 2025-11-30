import gradio as gr
from typing import List, Tuple

from .asr_tts import ASRWrapper, TTSWrapper
from .rag_agent import MedicalRAGAgent


asr = ASRWrapper()
tts = TTSWrapper()
agent = MedicalRAGAgent()


def chat_text(user_message: str, history: List[Tuple[str, str]]):
    if history is None:
        history = []

    if not user_message.strip():
        return "", history

    reply = agent.generate_response(user_message, history)
    history.append((user_message, reply))
    return "", history


def chat_speech(audio, history: List[Tuple[str, str]]):
    if history is None:
        history = []

    if audio is None:
        return "No audio detected.", None, history

    # 1. Speech -> text
    user_text = asr.speech_to_text(audio)

    # 2. RAG + LLM
    reply = agent.generate_response(user_text, history)
    history.append((user_text, reply))

    # 3. Text -> speech
    sr, audio_out = tts.text_to_speech(reply)

    return reply, (sr, audio_out), history


def build_app():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🩺 Medical Speech-to-Speech RAG Agent (Demo)

            This demo shows a **doctor–patient style assistant** with:

            - Speech ↔ Speech conversation
            - RAG over a small medical corpus
            - Multi-turn memory
            - Strong safety framing (not a real doctor)

            ⚠️ For educational use only. Do not rely on this for real medical decisions.
            """
        )

        with gr.Tab("Speech ↔ Speech"):
            state = gr.State([])  # multi-turn history

            audio_in = gr.Audio(source="microphone", type="numpy", label="Speak your symptoms")
            with gr.Row():
                text_out = gr.Textbox(label="Agent Response (Text)", lines=4)
                audio_out = gr.Audio(label="Agent Response (Audio)")

            audio_in.stop_recording(
                fn=chat_speech,
                inputs=[audio_in, state],
                outputs=[text_out, audio_out, state],
            )

        with gr.Tab("Text Chat"):
            state2 = gr.State([])
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Describe your symptoms", placeholder="I have a headache and mild fever...")

            def _chat_proxy(message, history):
                new_msg, new_hist = chat_text(message, history)
                return new_msg, new_hist

            msg.submit(_chat_proxy, [msg, chatbot], [msg, chatbot])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()