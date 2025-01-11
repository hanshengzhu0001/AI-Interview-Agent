from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class ActionAskTechnicalQuestion(Action):
    def name(self) -> str:
        return "action_ask_technical_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Example question or logic for asking technical questions
        dispatcher.utter_message(text="Can you explain how you would implement a binary search algorithm?")
        return []
