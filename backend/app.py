from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from rasa.core.agent import Agent
from rasa.core.utils import EndpointConfig
import speech_recognition as sr
from gtts import gTTS
import os

# Initialize Flask app
app = Flask(__name__)

# Load FLAN-T5 model and tokenizer for question generation
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Load DistilBERT model for intent recognition
distilbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Initialize Rasa agent (replace with your Rasa model path)
rasa_agent = Agent.load('rasa/models/your_rasa_model')

# Define a function to generate personalized questions based on the resume and job posting using FLAN-T5
def generate_personalized_questions(resume_text, job_posting_text):
    input_text = f"Resume: {resume_text} \nJob Posting: {job_posting_text} \nGenerate 3 personalized interview questions."
    
    inputs = flan_t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = flan_t5_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    
    question = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Define a function to classify the intent of the candidate's response using DistilBERT
def classify_intent(response_text):
    inputs = distilbert_tokenizer(response_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = distilbert_model(**inputs)
    prediction = outputs.logits.argmax().item()  # Take the class with the highest score
    return prediction  # This will return an integer corresponding to the predicted intent

# Rasa interaction API: send a message to the Rasa agent and receive a response
@app.route("/interact", methods=["POST"])
def interact_with_rasa():
    user_message = request.json.get("message")
    response = rasa_agent.handle_text(user_message)  # Send text to Rasa agent
    return jsonify({"response": response})

# API to generate personalized interview questions
@app.route("/generate_question", methods=["POST"])
def generate_question():
    # Receive the job posting and candidate's resume from the request
    data = request.get_json()
    resume_text = data.get('resume')
    job_posting_text = data.get('job_posting')

    # Generate questions based on the resume and job posting
    generated_question = generate_personalized_questions(resume_text, job_posting_text)

    # Return the generated question
    return jsonify({"question": generated_question})

# API to handle speech recognition (if needed for voice input)
@app.route("/transcribe_speech", methods=["POST"])
def transcribe_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Get audio file from request
    audio_file = request.files['audio']

    # Use speech recognition to transcribe the audio to text
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        # Recognize speech using Google Speech Recognition
        text = recognizer.recognize_google(audio)
        return jsonify({"transcription": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Speech recognition could not understand the audio"})
    except sr.RequestError:
        return jsonify({"error": "Could not request results from Google Speech Recognition"})

# API to convert text to speech (if you need the AI to speak)
@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    # Receive text from the request
    data = request.get_json()
    text = data.get('text')

    # Convert the text to speech using gTTS
    tts = gTTS(text, lang='en')
    audio_path = "output_audio.mp3"
    tts.save(audio_path)

    # Send the audio file back as a response
    return jsonify({"audio_file": audio_path})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)