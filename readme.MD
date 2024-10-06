# Birthday Party Planner AI

This project is a Flask-based web application that uses an AI agent to assist users in planning a birthday party. The AI helps users by recommending venues, asking for preferences, and handling other aspects of event planning. The AI is powered by Hugging Face's `flan-t5-large` model for generating recommendations and DuckDuckGo Search API to fetch venue options.

## Features

- **User Interaction**: Users provide their details, preferences, and location for the event.
- **Venue Search**: Uses DuckDuckGo Search to find venues based on the user's preferences and location.
- **AI Recommendations**: The AI agent suggests venues and helps with booking decisions using a custom prompt template and Hugging Face API.
- **Multi-step Conversations**: The user can interact with the AI in a back-and-forth conversation format.
- **Change Preferences**: Users can restart the process or change preferences at any time.

## Tech Stack

- **Flask**: Web framework used to build the application.
- **LangChain**: For managing the AI conversation.
- **Hugging Face API**: Provides the AI model (`flan-t5-large`) for processing user inputs.
- **DuckDuckGo Search API**: To fetch real-time venue suggestions.
- **HTML/CSS/JS**: For building the frontend and user interactions.

## Setup Instructions

### 1. Clone the Repository

```
bash
git clone https://github.com/your-repo/birthday-planner-ai.git
cd birthday-planner-ai
```
# 2. Install Dependencies
Make sure you have Python installed, then install the necessary packages using pip:

bash
Copy code
pip install -r requirements.txt
3. Set up Environment Variables
You will need a .env file with the following environment variables:

makefile
Copy code
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
Get your Hugging Face API key from Hugging Face.

4. Run the Application
Start the Flask application with:

bash
Copy code
python app.py
The app will be available at http://127.0.0.1:5000/.

5. Interaction
Open your browser and navigate to the index page.
Enter your details, preferences, and location.
Start chatting with the AI for venue suggestions or other party recommendations.
Notes
Rate Limiting
The DuckDuckGo Search API may sometimes hit rate limits. In case of such issues, the app retries the search after a short delay. If you're facing frequent rate limit issues, consider reducing the number of search results fetched or implementing a caching mechanism for queries.

Customization
You can easily customize the AI model by swapping out the Hugging Face model or adjusting the prompt template in party_planning_template. Additionally, venue search logic can be tweaked or expanded using other search APIs as needed.

Known Issues
Venues are only returned during the initial search or when the user explicitly asks for more venues.
The application might occasionally return slower responses due to the external APIs being used (Hugging Face or DuckDuckGo).