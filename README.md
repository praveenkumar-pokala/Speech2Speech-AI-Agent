# Medical Speech-to-Speech RAG Agent (Doctor–Patient, Multi-turn)

This repo contains a **low-latency speech-to-speech conversational RAG agent** for a doctor–patient style assistant:

- 🎙️ **Speech → Text** using Whisper (via `faster-whisper`)
- 📚 **RAG** over small medical knowledge snippets (easily replaceable with your own docs)
- 🧠 **LLM medical reasoning** (OpenAI `gpt-4o-mini` or similar)
- 🔁 **Full multi-turn memory** using conversation history
- 🔊 **Text → Speech** using F5-TTS (or replace with any TTS of your choice)
- 🧪 **Gradio UI** for speech-to-speech interaction
- 📓 Jupyter notebook for experimentation inside a notebook environment

> ⚠️ This project is for **educational & prototyping purposes only** and is **not a medical device**. It must **not** be used for real diagnosis or treatment. Always consult licensed clinicians.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

You may need system-level dependencies for `faster-whisper` and `f5-tts` (FFmpeg, etc.). If you have trouble with `f5-tts`, you can temporarily comment out the TTS parts and print text only.

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
# or on Windows
set OPENAI_API_KEY=sk-...
```

### 3. Run the Gradio app

```bash
python -m src.app
```

This will start a local Gradio server. Open the URL in your browser. You will see:

- A microphone input
- Transcribed user query
- Agent's textual answer
- Audio playback of the agent's answer

### 4. Use the Notebook

Open the notebook:

- `notebooks/medical_s2s_rag_agent.ipynb`

This shows the same pipeline step-by-step and runs Gradio **inline in the notebook** (e.g., Jupyter, Colab, VSCode).

---

## Repo Structure

```text
medical-s2s-rag-agent/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── rag_agent.py
│   ├── asr_tts.py
│   └── app.py
└── notebooks/
    └── medical_s2s_rag_agent.ipynb
```

### `src/config.py`

Central place for models, prompts, and basic hyperparameters.

### `src/asr_tts.py`

- Loads ASR model (Whisper via `faster-whisper`)
- Loads TTS (F5-TTS or any replacement)
- Helper functions for:
  - `speech_to_text(audio)`
  - `text_to_speech(text)`

### `src/rag_agent.py`

- Prepares and stores a tiny medical corpus
- Builds FAISS index
- Defines `MedicalRAGAgent` with methods:
  - `retrieve(query)` for RAG
  - `generate_response(query, history)` for multi-turn chat with RAG context

### `src/app.py`

- Wires ASR + RAG agent + TTS into a **single Gradio Blocks app**
- Maintains **multi-turn memory** via `gr.State`
- Exposes both:
  - Text chat interface
  - Speech-to-speech interface

---

## Safety and Disclaimer

This repository is a **technical demo**. It:

- May hallucinate or respond incorrectly
- Is **not** approved for medical use
- Should not be trusted for any real clinical decision

You **must** put clear disclaimers in front of any end users and ensure a licensed professional reviews any AI output.

---

Happy hacking 👨‍⚕️🧠🎙️