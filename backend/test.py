import requests

# URL for FLAN-T5 question generation endpoint
url = "http://127.0.0.1:5000/generate_question"  # Ensure this matches your Flask route for FLAN-T5

# Define input message for FLAN-T5
input_text = {
    "input_text": """
    I am applying for a Software Engineer position at your company. 
    The role requires expertise in Python, machine learning, cloud computing, 
    and scalable systems. Can you give me some technical interview questions for preparation?
    """
}

# Send POST request to the Flask app
response = requests.post(url, json=input_text)

# Print the response from the Flask app
if response.status_code == 200:
    questions = response.json()['question'].split('?')
    # Remove any empty strings or whitespace from the list
    questions = [q.strip() + '?' for q in questions if q.strip()]
    # Remove duplicates by converting to a set, then back to a list
    unique_questions = list(set(questions))
    
    # Print the unique questions
    for i, question in enumerate(unique_questions, 1):
        print(f"Question {i}: {question}")
else:
    print(f"Error: {response.status_code}, {response.text}")
