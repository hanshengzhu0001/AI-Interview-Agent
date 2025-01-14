import os
import asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from rasa.core.agent import Agent
from rasa.core.utils import EndpointConfig
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Upload directory for temporary audio storage
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INPUT_AUDIO_PATH = os.path.join(UPLOAD_FOLDER, "input_audio.wav")
OUTPUT_AUDIO_PATH = os.path.join(UPLOAD_FOLDER, "response.mp3")

# Initialize thread pool executor
executor = ThreadPoolExecutor(max_workers=4)

# Load FLAN-T5 for question generation
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/flan_t5_interview_agent")
flan_t5_tokenizer = AutoTokenizer.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/flan_t5_interview_agent")

# Load DistilBERT for intent recognition
distilbert_model = AutoModelForSequenceClassification.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/finetuned_distilbert")
distilbert_tokenizer = AutoTokenizer.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/finetuned_distilbert")

# Load Rasa Agent
rasa_model_path = "/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/20250111-165506-solid-crescendo.tar.gz"
endpoint_config = EndpointConfig(url="http://0.0.0.0:5005/webhooks/rest/webhook")
agent = Agent.load(rasa_model_path)

# Global variable to store the latest job posting
latest_job_posting = ""

# Function to classify user intent
def classify_intent(user_message: str):
    inputs = distilbert_tokenizer(user_message, return_tensors="pt", padding=True, truncation=True)
    outputs = distilbert_model(**inputs)
    logits = outputs.logits
    intent_idx = logits.argmax(dim=1).item()

    print("Logits:", logits)
    
    # Return intent label based on index
    intent_labels = ['greet', 'ask_company_profile', 'ask_resume', 'ask_technical_question', 'ask_behavioral_question', 'save_job_posting', 'goodbye']
    print(f"Classified Intent: {intent_labels[intent_idx]}")
    
    return intent_labels[intent_idx]

# Function to transcribe audio input
def transcribe_voice(file_path):
    recognizer = sr.Recognizer()

    # Ensure the file is in PCM WAV format
    try:
        audio = AudioSegment.from_file(file_path)
        audio.export(INPUT_AUDIO_PATH, format="wav")  # Overwrite input audio
    except Exception as e:
        return f"Error processing audio file: {e}"

    # Process the converted file with speech recognition
    with sr.AudioFile(INPUT_AUDIO_PATH) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}"

# Function to log conversation
def log_conversation(intent, user_message, response):
    with open("conversation_log.txt", "a") as log_file:
        log_file.write(f"Intent: {intent}\nUser Input: {user_message}\nAI Response: {response}\n\n")

# Function to handle user requests
def handle_request(user_message, intent):
    global latest_job_posting
    response = ""

    # Handle the 'save_job_posting' intent
    if intent == "save_job_posting":
        latest_job_posting = user_message
        with open("job_posting.txt", "w") as file:
            file.write(latest_job_posting)
        response = "Job posting saved successfully!"

    # Handle technical and behavioral question intents
    elif intent in ["ask_technical_question", "ask_behavioral_question"]:
        job_posting = ""
        if os.path.exists("job_posting.txt"):
            with open("job_posting.txt", "r") as file:
                job_posting = file.read().strip()

        if intent == "ask_technical_question":
            prompt = f"Generate a technical interview question related to the following job posting: {job_posting}" if job_posting else "Generate a technical interview question related to machine learning."
        elif intent == "ask_behavioral_question":
            prompt = f"Generate a behavioral interview question related to the following job posting: {job_posting}" if job_posting else "Generate a behavioral interview question related to machine learning."

        # Generate the question using FLAN-T5
        inputs = flan_t5_tokenizer(prompt, return_tensors="pt")
        outputs = flan_t5_model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        response = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Handle other intents via Rasa
    else:
        try:
            responses = asyncio.run(agent.handle_text(user_message))
            response = responses[0]["text"] if responses else "Sorry, I didn't understand that."
        except Exception as e:
            response = f"Error: {str(e)}"

    # Generate TTS audio for the response
    tts = gTTS(text=response, lang="en")
    tts.save(OUTPUT_AUDIO_PATH)  # Overwrite the previous audio response

    # Log the conversation
    log_conversation(intent, user_message, response)

    return {"response": response, "audio_path": "/audio/response.mp3"}  # Return both text and audio path


# Route to handle messages
@app.route('/rasa', methods=['POST'])
def rasa_route():
    user_message = None
    voice_file = None
    transcription = None

    # Check if the request contains JSON or form-data
    if request.is_json:
        user_message = request.json.get('message', "")
    else:
        user_message = request.form.get('message', "")
        voice_file = request.files.get('voice')

    # Handle voice input
    if voice_file:
        voice_file.save(INPUT_AUDIO_PATH)  # Save and overwrite input audio
        transcription = transcribe_voice(INPUT_AUDIO_PATH)
        user_message = transcription

    # Validate input
    if not user_message.strip():
        return jsonify({"error": "Empty input."}), 400

    intent = classify_intent(user_message)
    response = executor.submit(handle_request, user_message, intent).result()

    return jsonify({
        "response": response["response"],
        "audio_path": response["audio_path"],
        "transcription": transcription,
    })

# Route to serve audio files
@app.route('/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/mpeg")
    return jsonify({"error": "File not found."}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)