import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List, Dict
from duckduckgo_search import DDGS  # DuckDuckGo search for finding venues
import time
from duckduckgo_search.exceptions import RatelimitException

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Your Hugging Face API token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the Hugging Face API URL for the google/flan-t5-large model
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

# Define a custom LLM class to interact with Hugging Face API
class flanLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "inputs": prompt
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        response_json = response.json()

        if "error" in response_json:
            raise ValueError(f"Error from Hugging Face API: {response_json['error']}")

        # Check if the response is a list and handle accordingly
        if isinstance(response_json, list) and len(response_json) > 0:
            return response_json[0].get('generated_text', 'No answer found.')
        else:
            return 'No answer found.'

    @property
    def _identifying_params(self) -> dict:
        return {"model": "google/flan-t5-large"}

    @property
    def _llm_type(self) -> str:
        return "flan"

# Define a general prompt template for user interactions
party_planning_template = """You are an AI assistant helping a user plan a birthday party.

User details: {user_details}
Party preferences: {preferences}
Location: {location}
Venues found: {venues}

Conversation so far:
{conversation_history}

User's current input: "{user_input}"

Based on this, continue assisting the user.
"""

prompt = PromptTemplate(
    input_variables=["user_details", "preferences", "location", "venues", "conversation_history", "user_input"],
    template=party_planning_template,
)

# Initialize the custom flan LLM
llm = flanLLM()

# Function to use DuckDuckGo search to find venues based on the user's location
def search_venues_on_duckduckgo(location: str, preferences: str) -> List[str]:
    search_query = f"best venues for {preferences} birthday party in {location}"
    try:
        results = DDGS().text(search_query, max_results=5)
    except RatelimitException as e:
        print("Rate limit hit. Retrying after delay...")
        time.sleep(10)
        results = DDGS().text(search_query, max_results=5)

    if results:
        return [f"{result['title']}: {result['href']}" for result in results]
    else:
        return ["No venues found."]

# Function to get venue recommendations from the LLM
def get_recommended_venue(plan: Dict[str, str], venues: List[str], prompt: PromptTemplate, conversation_history: str, user_input: str) -> str:
    # Add the venues and conversation history to the plan
    plan["venues"] = "\n".join(venues)
    plan["conversation_history"] = conversation_history
    plan["user_input"] = user_input

    # Prepare the prompt for generating a recommendation or continuing the conversation
    formatted_prompt = prompt.format(**plan)

    # Ask the LLM to recommend the best venue and continue the conversation
    response = llm.invoke(formatted_prompt)
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_details = request.form['user_details']
        preferences = request.form['preferences']
        location = request.form['location']

        # Gather the initial plan details
        plan = {
            "user_details": user_details,
            "preferences": preferences,
            "location": location
        }

        # Get the venue search results
        venues = search_venues_on_duckduckgo(location, preferences)

        return render_template('chat.html', 
                               user_details=user_details, 
                               preferences=preferences, 
                               location=location, 
                               venues=venues)

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_details = request.form['user_details']
    preferences = request.form['preferences']
    location = request.form['location']
    user_input = request.form['user_input'].lower()  # Convert to lowercase to make command recognition easier
    conversation_history = request.form['conversation_history']

    # Initialize plan
    plan = {
        "user_details": user_details,
        "preferences": preferences,
        "location": location
    }

    # Check for booking requests
    if "book venue" in user_input:
        # Mock booking process (in a real-world app, you'd handle this differently)
        booking_message = "You have successfully booked the venue!"
        conversation_history += f"User: {user_input}\nAI: {booking_message}\n"
        return jsonify({
            "recommendation": booking_message,
            "conversation_history": conversation_history
        })
    
    # Check for requests to change preferences
    elif "change preferences" in user_input or "start over" in user_input:
        # Reset the flow and take the user back to the first questions
        restart_message = "Sure! Let's start over. Who is the party for, and what are your new preferences?"
        return jsonify({
            "recommendation": restart_message,
            "conversation_history": conversation_history
        })

    # Casual conversation
    elif "casual chat" in user_input:
        casual_message = "I'm happy to chat! Ask me anything or let me know if you need more party recommendations."
        conversation_history += f"User: {user_input}\nAI: {casual_message}\n"
        return jsonify({
            "recommendation": casual_message,
            "conversation_history": conversation_history
        })

    # Get venue search results again if needed
    venues = search_venues_on_duckduckgo(location, preferences)

    # Get LLM recommendation
    recommendation = get_recommended_venue(plan, venues, prompt, conversation_history, user_input)

    # Update the conversation history
    conversation_history += f"User: {user_input}\nAI: {recommendation}\n"

    return jsonify({
        "recommendation": recommendation,
        "conversation_history": conversation_history
    })


if __name__ == "__main__":
    app.run(debug=True)
