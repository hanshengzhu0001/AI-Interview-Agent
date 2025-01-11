from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class ActionAskResume(Action):
    def name(self) -> str:
        return "action_ask_resume"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Logic to ask the resume-related question
        dispatcher.utter_message(text="Can you share some highlights from your resume?")
        return []
