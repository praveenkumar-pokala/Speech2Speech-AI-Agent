import os

# OpenAI model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a careful, empathetic medical assistant.
"
    "You do NOT provide definitive diagnoses.
"
    "You help patients understand their symptoms, ask follow-up questions,
"
    "and encourage them to consult a qualified doctor for medical decisions.
"
    "Be concise, clear, and reassuring."
)

# ASR / TTS model names (can be changed by the user)
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "small")  # faster-whisper model name
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Sample corpus (for demo)
SAMPLE_DOCS = [
    "Fever, dry cough, and tiredness are common symptoms of many viral infections, "
    "including flu and COVID-19. Shortness of breath or chest pain may require urgent evaluation.",

    "Headache with nausea, sensitivity to light, or visual aura may suggest migraine, "
    "but serious causes like stroke must always be ruled out by a doctor.",

    "Chest pain that is heavy, squeezing, or radiates to the left arm or jaw can be a sign "
    "of a heart attack. This is a medical emergency and needs immediate care.",

    "For chronic conditions like diabetes or hypertension, regular follow-up with a physician, "
    "adherence to medications, and lifestyle changes are essential.",

    "Online information can help patients prepare better questions for their doctors, "
    "but it should never replace professional medical advice."
]