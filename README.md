# AI Interview Agent

An AI-powered interview assistant that dynamically generates and asks interview questions based on job descriptions, evaluates responses, and provides a seamless conversational experience. This project integrates **Flask**, **Rasa**, **Hugging Face Transformers**, and **gTTS** for a complete interactive interview system.

## Features

- **AI-Powered Question Generation**: Uses FLAN-T5 to generate technical and behavioral interview questions.  
- **Intent Classification**: Utilizes a fine-tuned DistilBERT model to classify user inputs.  
- **Conversational AI**: Rasa-powered chatbot handles user interactions.  
- **Speech-to-Text & Text-to-Speech**: Converts audio responses using Google Speech Recognition and gTTS.  
- **Question Handling**: Supports moving to the next question and storing question history.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hanshengzhu0001/AI-Interview-Agent.git
cd AI-Interview-Agent

```markdown
### 2. Set Up the Backend

**Install dependencies**  
Make sure you have Python 3.8+ installed. Then, install the required packages:
```

```bash
cd backend
pip install -r requirements.txt
```

```markdown
Start the Flask Server
```

```bash
python app.py
```

```markdown
### 3. Set Up the Frontend

Make sure you have Node.js and npm installed.
```

```bash
cd frontend
npm install
npm start
```

```markdown
### 4. Start the Rasa Model

Ensure Rasa is installed and properly trained. To start the Rasa chatbot:
```

```bash
cd rasa
rasa run
```

```markdown
Method | Endpoint | Description
------ | -------- | -----------
POST | /rasa | Sends user input (text/audio) and receives AI-generated response.
GET | /audio/<filename> | Retrieves the generated audio response.
GET | /conversation-history | Fetches the full conversation history.

## Usage
**Provide a Job Description**  
Send a job description to save relevant context for interview questions.

**Generate Interview Questions**  
Request technical or behavioral questions using predefined intents.

**Answer and Move to the Next Question**  
The AI will evaluate responses and seamlessly transition to the next question.

**Text or Voice Interaction**  
Users can either type responses or submit audio messages.
```
