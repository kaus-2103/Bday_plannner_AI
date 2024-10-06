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
git clone https://github.com/kaus-2103/Bday_plannner_AI.git
cd Bday_plannner_AI
```
# 2. Install Dependencies
Make sure you have Python installed, then install the necessary packages using pip:
```
pip install -r requirements.txt
```
# 3. Set up Environment Variables
You will need a .env file with the following environment variables:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```
Get your Hugging Face API key from Hugging Face.

# 4. Run the Application
Start the Flask application with:

```
python -m flask run
```
The app will be available at http://127.0.0.1:5000/.

# 5. Interaction
- Open your browser and navigate to the index page.
- Enter your details, preferences, and location.
- Start chatting with the AI for venue suggestions or other party recommendations.
- Interactions are hardcoded 

    1. ## More venues to get more venues 
    2. ## Change preferences to reroute to the index page
    3. ## Book to get booking of the venues ( Just simple reply now. Will implement web scraping later)
    4. ## Recommend a venue to get recommendation

# Birthday Party Planner AI - Dockerized Flask Application

This project is a Flask-based AI assistant for planning birthday parties. The AI helps users choose venues and make other party arrangements. This `README` explains how to use Docker to build, run, and push the image to Docker Hub.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Setup](#docker-setup)
  - [Build the Docker Image](#build-the-docker-image)
  - [Run the Docker Container](#run-the-docker-container)
  - [Push to Docker Hub](#push-to-docker-hub)
- [Access the Application](#access-the-application)

---

## Prerequisites

Before using Docker, ensure you have the following installed on your machine:

- [Docker](https://www.docker.com/products/docker-desktop) (v20+)
- [Docker Hub Account](https://hub.docker.com/)

---

## Docker Setup

### 1. Build the Docker Image

To build the Docker image, ensure you have the `Dockerfile` in your project root directory.

```bash
docker build -t your_dockerhub_username/birthday-planner-ai .
```

Replace `your_dockerhub_username` with your actual Docker Hub username.

### 2. Run the Docker Container

Once the image is built, you can run the container using:

```bash
docker run -p 5000:5000 your_dockerhub_username/birthday-planner-ai
```

This will start the application on port 5000. The Flask app will be accessible via `http://localhost:5000`.

### 3. Push to Docker Hub

If you want to make the image publicly available, you can push the Docker image to Docker Hub:

# 1. Log in to Docker Hub:

    ```bash
    docker login
    ```

    Enter your Docker Hub credentials when prompted.

# 2. Push the Docker image:

    ```bash
    docker push your_dockerhub_username/birthday-planner-ai
    ```

# Now, your image will be available on Docker Hub, and anyone can pull and run it using the following commands:

```bash
docker pull your_dockerhub_username/birthday-planner-ai
docker run -p 5000:5000 your_dockerhub_username/birthday-planner-ai
```

---

## Access the Application

After running the container, you can access the AI assistant on `http://localhost:5000`. Use the interface to provide the necessary details for the birthday party (user details, preferences, location) and interact with the AI to find venues and plan the event.

---

## Environment Variables

The app uses the following environment variables, which should be set in the `.env` file:

- `HUGGINGFACEHUB_API_TOKEN`: API token for Hugging Face.
- Any other tokens or credentials required for external APIs (e.g., DuckDuckGo search).

Make sure to include your `.env` file with the necessary variables in the Docker build if needed, or manage secrets securely.

---

## Troubleshooting

1. **Rate Limits**: If you run into rate limits with DuckDuckGo API, the app will automatically retry with exponential backoff.
2. **Environment Variables**: Ensure all required environment variables are provided in a `.env` file or through Docker secrets.
3. **Docker Hub Push Issues**: If you encounter problems pushing the image to Docker Hub, ensure you're logged in (`docker login`) and have tagged the image correctly.

---

## Conclusion

With Docker, you can easily build, run, and share the Birthday Party Planner AI application, allowing for consistent deployment across different environments.

# Notes
- Rate Limiting
The DuckDuckGo Search API may sometimes hit rate limits. In case of such issues, the app retries the search after a short delay. If you're facing frequent rate limit issues, consider reducing the number of search results fetched or implementing a caching mechanism for queries.

- Model Loading
Sometimes Models takes time to load in Hugging Face. 

Live Demo: https://tinyurl.com/bdfxz4zb 
Note: It takes a lot of time to respond for API issues and gives internal server error mostly and also it is on render so it takes 50 seconds to start up. 
