import os
import uuid
import asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from rasa.core.agent import Agent
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Directory configurations
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

QUESTION_LIST_FILE = "question_list.txt"
CONVERSATION_LOG_FILE = "conversation_log.txt"

# Thread pool executor
executor = ThreadPoolExecutor(max_workers=4)

# Load FLAN-T5 for question generation
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/flan_t5_interview_agent")
flan_t5_tokenizer = AutoTokenizer.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/flan_t5_interview_agent")

# Load DistilBERT for intent recognition
distilbert_model = AutoModelForSequenceClassification.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/finetuned_distilbert")
distilbert_tokenizer = AutoTokenizer.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/finetuned_distilbert")

# Load Rasa model
rasa_model_path = "/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/20250111-165506-solid-crescendo.tar.gz"
agent = Agent.load(rasa_model_path)

# Globals
latest_job_posting = ""
conversation_history = []  # To store conversation with timestamps

# Function to classify intent using DistilBERT
def classify_intent(user_message: str):
    """
    Classifies the user's intent using the fine-tuned DistilBERT model.
    """
    inputs = distilbert_tokenizer(user_message, return_tensors="pt", padding=True, truncation=True)
    outputs = distilbert_model(**inputs)
    intent_idx = outputs.logits.argmax(dim=1).item()
    intent_labels = [
        'greet',            # 0
        'ask_company_profile',  # 1
        'ask_resume',       # 2
        'get_technical',    # 3 (formerly ask_technical_question)
        'get_behavioral',   # 4 (formerly ask_behavioral_question)
        'next_question',    # 5
        'save_job_posting', # 6
        'goodbye'           # 7
    ]
    intent = intent_labels[intent_idx]
    
    print(f"Classified Intent: {intent}")
    return intent

# Function to transcribe voice input
def transcribe_voice(file_path):
    """
    Converts an audio file to a temporary WAV and uses SpeechRecognition to obtain transcription.
    """
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_file(file_path)
        temp_wav = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.wav")
        audio.export(temp_wav, format="wav")
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        return transcription, temp_wav
    except sr.UnknownValueError:
        return None, None  # Return an empty string instead of an error message
    except Exception as e:
        return f"Error processing audio: {e}", None

# Function to generate text-to-speech audio
def generate_tts_response(text):
    """
    Generates an MP3 file for the given text using gTTS and returns the file path.
    """
    audio_filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.mp3")
    tts = gTTS(text=text, lang="en")
    tts.save(audio_filename)
    return audio_filename

# Function to log conversations with timestamps
def log_conversation(intent, user_message, response, user_audio_path=None, response_audio_path=None):
    """
    Logs each interaction to both memory (conversation_history) and a file (conversation_log.txt).
    """
    timestamp = datetime.utcnow().isoformat()
    entry = {
        "timestamp": timestamp,
        "user_message": user_message,
        "machine_response": response,
        "intent": intent,
        "user_audio_path": user_audio_path,
        "response_audio_path": response_audio_path
    }
    
    # In-memory history
    conversation_history.append(entry)
    
    # Append to conversation_log.txt
    with open(CONVERSATION_LOG_FILE, "a") as log_file:
        log_file.write(
            f"{timestamp}\n"
            f"User: {user_message}\n"
            f"AI: {response}\n"
            f"Intent: {intent}\n\n"
        )

def get_next_question():
    """
    Retrieves the next question from question_list.txt (if it exists), 
    then removes it from the file. 
    """
    if not os.path.exists(QUESTION_LIST_FILE):
        return "No questions available. Please generate questions first."

    with open(QUESTION_LIST_FILE, "r") as file:
        questions = file.readlines()

    if not questions:
        return "No more questions left in the list."

    next_question = questions.pop(0).strip()

    # Rewrite the file with remaining questions
    with open(QUESTION_LIST_FILE, "w") as file:
        file.writelines(questions)

    return next_question

# Core function to handle user requests
def handle_request(user_message, intent):
    """
    Main logic that decides how to respond based on the user's intent.
    """
    global latest_job_posting
    response_text = ""

    # Generate a unique audio filename for the response
    response_audio_filename = f"{uuid.uuid4().hex}.mp3"
    response_audio_path = os.path.join(UPLOAD_FOLDER, response_audio_filename)

    if intent == "save_job_posting":
        # Save the latest job posting content
        latest_job_posting = user_message
        with open("job_posting.txt", "w") as file:
            file.write(latest_job_posting)
        response_text = "Job posting saved successfully!"

    elif intent in ["get_technical", "get_behavioral"]:
        # Generate a set of questions using FLAN-T5, save them to question_list.txt
        job_posting = ""
        if os.path.exists("job_posting.txt"):
            with open("job_posting.txt", "r") as file:
                job_posting = file.read().strip()

        prompt_type = "technical" if intent == "get_technical" else "behavioral"
        prompt = (
            f"Generate a set of {prompt_type} interview questions "
            f"related to: {job_posting or 'machine learning'}"
        )
        inputs = flan_t5_tokenizer(prompt, return_tensors="pt")
        outputs = flan_t5_model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # **Fix: Ensure proper question separation by splitting at "?"**
        full_text = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        questions = [q.strip() + "?" for q in full_text.split("?") if q.strip()]

        # Ensure we only output the FIRST question
        question = questions[0] if questions else "No questions were generated."

        # Save all questions to the file
        with open(QUESTION_LIST_FILE, "w") as file:
            file.writelines([q + "\n" for q in questions])
        
        first_question = get_next_question()

        response_text = f"Here is your first question: {first_question}"

    elif intent == "next_question":
        # Pull the next question from question_list.txt
        response_text = get_next_question()

    else:
        # Fallback: pass user_message to Rasa
        try:
            rasa_responses = asyncio.run(agent.handle_text(user_message))
            response_text = rasa_responses[0]["text"] if rasa_responses else "Sorry, I couldn't understand that."
        except Exception as e:
            response_text = f"Error: {str(e)}"

    # Generate TTS for the response
    tts_path = generate_tts_response(response_text)

    # Log conversation
    log_conversation(
        intent=intent,
        user_message=user_message,
        response=response_text,
        user_audio_path=None,
        response_audio_path=f"/audio/{os.path.basename(tts_path)}"
    )

    return {
        "response": response_text,
        "audio_path": f"/audio/{os.path.basename(tts_path)}"
    }

# Endpoint for handling Rasa requests
@app.route('/rasa', methods=['POST'])
def rasa_route():
    user_message = None
    transcription = None
    user_audio_path = None

    if request.is_json:
        user_message = request.json.get('message', "")
    else:
        user_message = request.form.get('message', "")
        voice_file = request.files.get('voice')

        if voice_file:
            unique_id = uuid.uuid4().hex
            user_audio_filename = f"{unique_id}.wav"
            user_audio_path = os.path.join(UPLOAD_FOLDER, user_audio_filename)
            voice_file.save(user_audio_path)

            text_result, actual_path = transcribe_voice(user_audio_path)

            # **Fix: Ignore failed transcriptions**
            if text_result:  # Only use transcription if it's non-empty
                user_message = text_result
            else:
                return jsonify({
                    "response": "I couldn't understand your audio. Please try again.",
                    "audio_path": None,
                    "transcription": None,
                }), 400

    # Validate input
    if not user_message.strip():
        return jsonify({
            "response": "No valid input detected. Please provide a message or try again.",
            "audio_path": None,
            "transcription": transcription,
        }), 400

    # Classify and handle
    intent = classify_intent(user_message)
    response = executor.submit(handle_request, user_message, intent).result()

    return jsonify({
        "response": response["response"],
        "audio_path": response["audio_path"],
        "transcription": user_message,
        "user_audio_path": f"/audio/{os.path.basename(user_audio_path)}" if user_audio_path else None
    })

# Endpoint to serve audio files
@app.route('/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/mpeg")
    return jsonify({"error": "File not found."}), 404

# Endpoint to retrieve conversation history
@app.route('/conversation-history', methods=['GET'])
def get_conversation_history():
    return jsonify({"conversation_history": conversation_history})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
