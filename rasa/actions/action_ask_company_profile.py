from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import random

class ActionGenerateCompanyProfileQuestion(Action):

    def name(self) -> str:
        return "action_generate_company_profile_question"

    def run(self, dispatcher, tracker, domain):
        # Predefined company profile-related questions
        questions = [
            "Can you tell me why you are interested in working for this company?",
            "What excites you the most about the position you are applying for?",
            "How do you see yourself contributing to our company’s mission?",
            "What do you know about our company's values and culture?",
            "What aspects of our work environment or culture are you most drawn to?",
            "How do you think your experience aligns with the goals of our company?",
            "Can you describe what attracts you to our company’s vision and future goals?"
        ]
        
        # Randomly select a question to ask
        question = random.choice(questions)
        
        # Send the question to the candidate
        dispatcher.utter_message(text=question)
        
        return []
