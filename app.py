import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from google import genai
from google.genai.errors import APIError

# Load environment variables
load_dotenv()

# Initialize Flask app with proper static folder
app = Flask(__name__)
CORS(app)

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
model = None

if GEMINI_API_KEY:
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)
        model = "gemini-2.5-flash"  # Stable model with better rate limits
        print("✓ Gemini API initialized successfully")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        GEMINI_API_KEY = None
else:
    print("Warning: GEMINI_API_KEY not found in environment variables!")

# --- Flask Routes ---


@app.route("/")
def index():
    """Serve the main HTML page"""
    return render_template("index.html")


def preprocess_question(question):
    """
    Preprocess the input question:
    - Lowercasing
    - Tokenization
    - Punctuation removal
    
    Args:
        question (str): Raw input question
        
    Returns:
        dict: Preprocessing steps and processed question
    """
    import string
    
    # Store original
    original = question
    
    # Convert to lowercase
    question_lower = question.lower()
    
    # Remove punctuation
    question_no_punct = question_lower.translate(
        str.maketrans('', '', string.punctuation)
    )
    
    # Tokenization (split into words)
    tokens = question_no_punct.split()
    
    # Join tokens back
    processed = ' '.join(tokens)
    
    return {
        "original": original,
        "lowercased": question_lower,
        "punctuation_removed": question_no_punct,
        "tokens": tokens,
        "processed": processed
    }


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    API endpoint to receive questions and return AI-generated answers
    """
    if not client or not model:
        return (
            jsonify(
                {"error": "Gemini service not configured. Please check your API key."}
            ),
            500,
        )

    try:
        # Get question from request
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data["question"].strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Preprocess the question
        preprocessing = preprocess_question(question)

        # Generate response using Google Gemini AI
        response = client.models.generate_content(model=model, contents=question)

        # Extract answer from response
        if response and response.text:
            answer = response.text
        else:
            answer = "I couldn't generate a response. Please try again."

        return (
            jsonify({
                "question": question,
                "answer": answer,
                "preprocessing": preprocessing,
                "status": "success"
            }),
            200,
        )

    except APIError as e:
        # Specific error handling for API issues (rate limits, invalid key, etc.)
        app.logger.error(f"Gemini API Error: {str(e)}")
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


def main():
    """Run the Flask application"""
    # ... (Rest of the main function remains the same)
    print("=" * 60)
    print("AI Question Answering Interface")
    print("=" * 60)
    print("Server starting...")

    if not GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY not found!")
        print("Please create a .env file with your Google Gemini API key:")
        print("GEMINI_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        print("=" * 60)

    print("\n✓ Server running at: http://localhost:5000")
    print("✓ Open your browser and navigate to the URL above")
    print("\nPress Ctrl+C to stop the server\n")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
