# -------------------------------------------------------------------------
# Stress Detection Chatbot Backend (Flask)
# Uses Logistic Regression for stress detection
# Uses LOCAL Ollama Server (mistral:latest) for empathetic responses
# -------------------------------------------------------------------------

import re
import random
from flask import Flask, request, jsonify, render_template
import requests
import joblib

# --- Configuration for Ollama Local Inference ---
MODEL_NAME = "mistral:latest"  # Match your installed model name
API_URL = "http://localhost:11434/api/chat"

# --- Flask App Setup ---
app = Flask(__name__)

# --- Load Trained Logistic Regression Model and TF-IDF Vectorizer ---
try:
    stress_model = joblib.load("logistic_stress_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Loaded Logistic Regression model and TF-IDF vectorizer successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load model or vectorizer: {e}")
    stress_model = None
    vectorizer = None

# --- Context Memory (short-term conversation tracking) ---
last_was_stress = False

# -------------------------------------------------------------------------
# Function: Predict Stress Level using Logistic Regression
# -------------------------------------------------------------------------
def predict_stress_level(text):
    """Predicts stress level using Logistic Regression model and TF-IDF vectorizer."""
    if not stress_model or not vectorizer:
        # Fallback keyword detection if model files are missing
        keywords = ["stress", "anxiety", "confused", "overwhelmed",
                    "burnt out", "panic", "tired", "deadline", "exam"]
        score = sum(text.lower().count(kw) for kw in keywords)
        return ("stress", 0.95) if score >= 1 else ("non-stress", 0.10)

    X_input = vectorizer.transform([text])
    prob_stress = stress_model.predict_proba(X_input)[0][1]
    label = "stress" if prob_stress >= 0.4 else "non-stress"
    return label, float(prob_stress)

# -------------------------------------------------------------------------
# Function: Generate concise, structured response using Mistral (Ollama)
# -------------------------------------------------------------------------
def generate_hf_response(user_message, max_items=3, max_chars=600):
    """
    Calls local Ollama API (Mistral 7B) and returns a concise, structured reply.
    - Prompts the model to return up to `max_items` short actionable steps (numbered).
    - Post-processes the model output to ensure it's concise and formatted.
    """
    system_prompt = (
        "You are an empathetic, supportive mental wellness coach. The user is stressed. "
        "Respond with a concise, structured, numbered list of actionable strategies (no more than "
        f"{max_items} items). Each item should be one short sentence (approx. 8-20 words). "
        "Start each item with '1.' '2.' etc. Do NOT add long introductions or extra paragraphs. "
        "Keep the response practical and focused."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        # reduce token prediction to encourage brevity
        "options": {"temperature": 0.6, "num_predict": 200},
        "stream": False
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Extract raw content (Ollama chat API format)
        raw = data.get("message", {}).get("content", "")
        if not raw:
            return "Sorry, the AI returned an empty response. Please try again."

        # Post-processing
        text = re.sub(r'\r\n', '\n', raw).strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # Try to extract numbered items first
        numbered_items = []
        for ln in lines:
            # matches "1. something" or "1) something" or "- 1. something" patterns
            m = re.match(r'^\s*(?:\d+[\.\)]|\-\s*\d+[\.\)])\s*(.*)$', ln)
            if m:
                numbered_items.append(m.group(1).strip())
            else:
                m2 = re.match(r'^\s*(\d+)\s+(.*)$', ln)
                if m2:
                    numbered_items.append(m2.group(2).strip())

        if numbered_items:
            items = numbered_items[:max_items]
        else:
            # Fallback: split into sentences
            sentences = re.split(r'(?<=[\.\?\!])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            items = sentences[:max_items]

        # Ensure items are reasonably short; truncate if wordy
        clean_items = []
        for it in items:
            words = it.split()
            if len(words) > 20:
                it = ' '.join(words[:20]).rstrip('.,;:') + '...'
            clean_items.append(it)

        # Build numbered reply
        numbered_reply = ""
        for idx, it in enumerate(clean_items, start=1):
            numbered_reply += f"{idx}. {it}\n"

        # Trim overall length if necessary
        if len(numbered_reply) > max_chars:
            allowed = numbered_reply[:max_chars]
            last_nl = allowed.rfind('\n')
            if last_nl > 0:
                numbered_reply = allowed[:last_nl].rstrip() + '\n'
            else:
                numbered_reply = allowed[:max_chars].rstrip() + '...'

        return numbered_reply.strip()

    except requests.exceptions.RequestException as e:
        print(f"[Ollama API Error] Could not connect to {API_URL}: {e}")
        return "Error: Could not connect to Ollama. Please make sure 'ollama serve' is running."
    except Exception as e:
        print(f"[Unexpected LLM Error] {e}")
        return "An internal error occurred while generating the supportive response."

# -------------------------------------------------------------------------
# Route: Home Page (Landing)
# -------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

# -------------------------------------------------------------------------
# Route: Chat Page
# -------------------------------------------------------------------------
@app.route("/chat")
def chat_page():
    return render_template("index.html")

# -------------------------------------------------------------------------
# Route: Main Chat Prediction
# -------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Handles stress detection and decides whether to use Mistral or motivational replies."""
    global last_was_stress

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"reply": "Invalid request."}), 400

    user_message = data["message"]

    # Default fallback (should get overwritten below)
    reply = "Internal System Error: reply not set."
    reply_type = "error"

    # Step 1: Predict stress probability
    label, prob_stress = predict_stress_level(user_message)

    is_currently_stressful = prob_stress >= 0.4
    use_llm = is_currently_stressful or last_was_stress

    if use_llm:
        # Use the LLM to generate a concise structured reply
        reply = generate_hf_response(user_message, max_items=3, max_chars=600)
        reply_type = "stress"
    else:
        # Use predefined motivational replies
        non_stress_replies = [
            "That sounds like a great plan! What's next for you?",
            "Thanks for sharing! What are your thoughts on that topic?",
            "I see. That's good to know!",
            "Got it! How are you feeling about that today?",
            "You’re doing well — keep it up!"
        ]
        reply = random.choice(non_stress_replies)
        reply_type = "non-stress"

    # Update conversation short-term context
    last_was_stress = is_currently_stressful

    return jsonify({
        "prediction": reply_type,
        "probability": round(prob_stress, 3),
        "reply_type": reply_type,
        "reply": reply,
        "threshold_explanation": "Messages with probability >= 0.4 are flagged as stress. LLM used for flagged messages or immediate continuation."
    })

# -------------------------------------------------------------------------
# Run Flask Application
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 Running Stress Relief Chatbot with LOCAL Ollama Model: {MODEL_NAME}")
    app.run(debug=True)
