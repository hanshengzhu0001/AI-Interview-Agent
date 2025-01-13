import asyncio
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to allow cross-origin requests from frontend
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from rasa.core.agent import Agent
from rasa.core.utils import EndpointConfig
import speech_recognition as sr
from gtts import gTTS
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow cross-origin requests from frontend
CORS(app)

# Load FLAN-T5 model and tokenizer for question generation
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/flan_t5_interview_agent")
flan_t5_tokenizer = AutoTokenizer.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/flan_t5_interview_agent")

# Load DistilBERT model and tokenizer for intent recognition
distilbert_model = AutoModelForSequenceClassification.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/finetuned_distilbert")
distilbert_tokenizer = AutoTokenizer.from_pretrained("/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/finetuned_distilbert")

# Load Rasa Agent for dialog management
rasa_model_path = "/Users/hanszhu/Desktop/AI_Interview_Agent/rasa/models/20250111-165506-solid-crescendo.tar.gz"
endpoint_config = EndpointConfig(url="http://0.0.0.0:5005/webhooks/rest/webhook")
agent = Agent.load(rasa_model_path)

# classify intent
def classify_intent(user_message: str):
    inputs = distilbert_tokenizer(user_message, return_tensors="pt", padding=True, truncation=True)
    outputs = distilbert_model(**inputs)
    logits = outputs.logits
    intent = logits.argmax(dim=1).item()

    print("Logits:", logits)
    
    # Return intent label (add your own logic for mapping logits to intent labels)
    intent_labels = ['greet', 'ask_company_profile', 'ask_resume', 'ask_technical_question', 'ask_behavioral_question', 'goodbye']
    print(f"Classified Intent: {intent_labels[intent]}")

    return intent_labels[intent]

# Route to handle messages from frontend
@app.route('/rasa', methods=['POST'])
async def rasa():
    data = request.get_json()
    user_message = data.get('message')

    # Step 1: Classify the intent using DistilBERT
    intent = classify_intent(user_message)

    # Step 2: Based on intent, either generate a question or call Rasa
    if intent == 'ask_technical_question' or intent == 'ask_behavioral_question':
            # Generate interview question using FLAN-T5
            prompt = ""
            if intent == 'ask_technical_question':
                prompt = "Generate a technical interview question related to machine learning."
            elif intent == 'ask_behavioral_question':
                prompt = "Generate a behavioral interview question related to machine learning."

            inputs = flan_t5_tokenizer(prompt, return_tensors="pt")
            output = flan_t5_model.generate(
                inputs.input_ids,
                max_new_tokens=50,          # Adjust this value as needed
                num_beams=5,                # For better quality responses
                no_repeat_ngram_size=2,     # Prevent repetition
                early_stopping=True
            )
            question = flan_t5_tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"Generated Question: {question}")

            # Return the generated question in the response as JSON
            return jsonify({"response": question})

    elif intent == 'greet':
        return jsonify({"response": "Hello! How can I help you today?"})
    elif intent == 'ask_company_profile':
        return jsonify({"response": "Our company is a leading innovator in AI technologies..."})
    elif intent == 'ask_resume':
        return jsonify({"response": "Can you tell me more about your experience with data science?"})
    elif intent == 'goodbye':
        return jsonify({"response": "Goodbye! Have a great day!"})

    # Step 3: If the intent is not recognized, pass it to Rasa for dialog management
    try:
        responses = await agent.handle_text(user_message)
        if responses:
            return jsonify({"response": responses[0]["text"]})
        else:
            return jsonify({"response": "Sorry, I didn't understand that."})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
