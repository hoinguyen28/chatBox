from flask import Flask, render_template, request, jsonify
from chatbot_model import ChatbotPhongChongLuaDao

app = Flask(__name__)
chatbot = ChatbotPhongChongLuaDao()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = chatbot.respond(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)