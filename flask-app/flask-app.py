from flask import Flask, render_template, request, jsonify
import pickle

# Load the chatbot model
with open('chatbot_model.pkl', 'rb') as f:
    chatbot = pickle.load(f)

# Create a Flask application
app = Flask(__name__)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the chatbot endpoint
@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    # Get the user input from the request data
    user_input = request.json['user_input']

    # Retrieve the response associated with the user input
    response = chatbot.respond(user_input)

    # Return the response as JSON
    return jsonify({'response': response})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)  # Set debug=True for development, you can change it to False for production
