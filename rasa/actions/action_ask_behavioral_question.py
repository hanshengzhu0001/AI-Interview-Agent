from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class ActionAskBehavioralQuestion(Action):
    def name(self) -> str:
        return "action_ask_behavioral_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Example question or logic for asking behavioral questions
        dispatcher.utter_message(text="Tell me about a time when you had to overcome a challenge at work.")
        return []
