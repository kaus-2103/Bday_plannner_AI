import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List, Dict
from duckduckgo_search import DDGS
import time
from duckduckgo_search.exceptions import RatelimitException

load_dotenv()

app = Flask(__name__)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

# Global variables to hold venues and current index
venues = []
current_venue_index = 0

class flanLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "inputs": prompt
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response_json = response.json()

        if "error" in response_json:
            raise ValueError(f"Error from Hugging Face API: {response_json['error']}")

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

llm = flanLLM()

def search_venues_on_duckduckgo(location: str, preferences: str) -> List[str]:
    search_query = f"best venues for {preferences} birthday party in {location}"
    retries = 3
    delay = 5 
    results = None  

    for attempt in range(retries):
        try:
            results = DDGS().text(search_query, max_results=10)
            break  
        except RatelimitException:
            print(f"Rate limit hit. Retrying in {delay} seconds...")
            time.sleep(delay)  # Wait before retrying
            delay *= 2  # Exponentially increase the delay

    if results:
        return [f"{result['title']}: {result['href']}" for result in results]
    else:
        return ["No venues found."]

def get_recommended_venue(plan: Dict[str, str], venues: List[str], prompt: PromptTemplate, conversation_history: str, user_input: str) -> str:
    plan["venues"] = "\n".join(venues)
    plan["conversation_history"] = conversation_history
    plan["user_input"] = user_input
    formatted_prompt = prompt.format(**plan)

    response = llm.invoke(formatted_prompt)
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    global venues, current_venue_index

    if request.method == 'POST':
        user_details = request.form['user_details']
        preferences = request.form['preferences']
        location = request.form['location']

        # Reset venues and current index on new request
        venues = search_venues_on_duckduckgo(location, preferences)
        current_venue_index = 0  # Reset index to start from the beginning

        # Display the first 5 venues
        venues_display = "\n".join(venues[:5])
        current_venue_index = 5  # Update the index after displaying the first set

        return render_template('chat.html', 
                               user_details=user_details, 
                               preferences=preferences, 
                               location=location, 
                               venues=venues_display)

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global venues, current_venue_index
    user_details = request.form['user_details']
    preferences = request.form['preferences']
    location = request.form['location']
    user_input = request.form['user_input'].lower()
    conversation_history = request.form['conversation_history']

    plan = {
        "user_details": user_details,
        "preferences": preferences,
        "location": location
    }

    # Check for booking requests
    if "book venue" in user_input:
        booking_message = "You have successfully booked the venue!"
        conversation_history += f"User: {user_input}\nAI: {booking_message}\n"
        return jsonify({
            "recommendation": booking_message,
            "conversation_history": conversation_history
        })

    # Check for requests to change preferences
    elif "change preferences" in user_input or "start over" in user_input:
        restart_message = "Sure! Let's start over. Redirecting you to the main page..."
        return jsonify({
            "recommendation": restart_message,
            "redirect": True,  # Indicate that this should trigger a redirect
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

    # Check if user wants more venues
    elif "more venues" in user_input:
        if current_venue_index < len(venues):
            venues_display = "\n".join(venues[current_venue_index:current_venue_index + 5])  
            current_venue_index += 5  
            conversation_history += f"User: {user_input}\nAI: Here are more venues:\n{venues_display}\n"
        else:
            venues_display = "No more venues available."
            conversation_history += f"User: {user_input}\nAI: {venues_display}\n"

        return jsonify({
            "recommendation": f"Here are more venues:\n{venues_display}",
            "conversation_history": conversation_history
        })

    # Get LLM recommendation without venues unless explicitly asked
    # Only return venues in the first venue search, or when "more venues" is requested
    if "venue" in user_input:
        recommendation = get_recommended_venue(plan, venues, prompt, conversation_history, user_input)
    else:
        recommendation = "I'm happy to chat! Ask me anything or let me know if you need more party recommendations."

    # Update conversation history with user input and AI recommendation
    conversation_history += f"User: {user_input}\nAI: {recommendation}\n"

    return jsonify({
        "recommendation": recommendation,
        "conversation_history": conversation_history
    })


if __name__ == "__main__":
    app.run(debug=False)
