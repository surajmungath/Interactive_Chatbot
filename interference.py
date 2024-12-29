import pickle
import random
import nltk

# Ensure nltk data is downloaded
nltk.download("punkt")

# Load the trained model and vectorizer
with open("chatbot_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Define intents (should match training data structure)
data = {
    "intents": [
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do","Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age","age","what is your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {"tag": "greetings",
     "patterns": ["Hello there", "Hey, How are you", "Hey", "Hi", "Hello", "Anybody", "Hey there"],
     "responses": ["Hello, I'm your helping bot", "Hey it's good to see you", "Hi there, how can I help you?"],
     "context": [""]
    },
    {"tag": "thanks",
     "patterns": ["Thanks for your quick response", "Thank you for providing the valuable information", "Awesome, thanks for helping"],
     "responses": ["Happy to help you", "Thanks for reaching out to me", "It's My pleasure to help you"],
     "context": [""]
    },
    {"tag": "no_answer",
     "patterns": [],
     "responses": ["Sorry, Could you repeat again", "provide me more info", "can't understand you"],
     "context": [""]
    },
    {"tag": "help",
     "patterns": ["What help you can do?", "What are the helps you provide?", "How you could help me",],
     "responses": [ "i can help you by generate new storys,play game with you,and also i can help you by giving song recommendation"],
     "context": [""]
    },
    {"tag": "goodbye",
        "patterns": ["bye bye", "Nice to chat with you", "Bye", "See you later buddy", "Goodbye"],
        "responses": [ "bye bye, thanks for reaching", "Have a nice day there", "See you later"],
        "context": [""]
    },
     {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "Hi there", "Morning", "Evening"],
        "responses": [
            "Hi, my name is Nava. How can I help you?",
            "Hello! I'm Nava. What can I do for you today?",
            "Hey there! How can I assist you?",
            "Hello! I'm Nava. How can I assist you today?",
            "Good morning! How can I help you?",
            "Good evening! What can I do for you today?"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you", "Goodbye", "Talk to you later", "I'm leaving", "Take care"],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Goodbye! If you need any more help, just ask.",
            "Sure, talk to you later! Have a good day!",
            "Alright, goodbye! Feel free to return if you need any help.",
            "You too! Bye!"
        ]
    },
    {
        "tag": "name",
        "patterns": ["Name", "your name", "what is your name"],
        "responses": ["My name is Nava."]
    },
    {
        "tag": "password",
        "patterns": ["password", "password reset", "forgot password"],
        "responses": ["To reset your password, go to the login page and click on 'Forgot Password'."]
    },
    {
        "tag": "location",
        "patterns": ["Where are you located","place","orgin","location"],
        "responses": ["We are based in KBMGCT Thalassery."]
    },
    {
        "tag": "support",
        "patterns": ["support", "contact","what is your contact number","contact number","your support"],
        "responses": [
            "You can contact support by emailing navachatbot@gmail.com or calling 7909192967."
        ]
    },
    {
        "tag": "hours",
        "patterns": ["What are your hours of operation?"],
        "responses": ["I am only available when my creator is online."]
    },
    {
        "tag": "wellbeing",
        "patterns": ["How are you"],
        "responses": ["I'm just a bot, but I'm here to help you!"]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like?", "Tell me about the weather"],
        "responses": [
            "I don't have live weather data, but you can check a weather website or app."
        ]
    },
    {
        "tag": "joke",
        "patterns": ["Tell me a joke","joke","joke please","tell me comedy","comedy","Can you tell me a joke?", "I want to hear a joke", "Make me laugh"],
        "responses": [
            "Why don't scientists trust atoms? Because they make up everything!😂😂",
            "Why don't scientists trust atoms? Because they make up everything!😂😂",
            "Why did the scarecrow win an award? Because he was outstanding in his field😂😂!",
            "Why don't skeletons fight each other? They don't have the guts!😂😂",
            "Why did the bicycle fall over? Because it was two-tired!😂😂",
            "What do you call cheese that isn't yours? Nacho cheese!😂😂"
        ]
    },
    {
        "tag": "preferences",
        "patterns": ["What's your favorite color?", "colour"],
        "responses": [
            "As a bot, I don't have preferences, but I think blue is quite nice!"
        ]
    },
    {
        "tag": "hobbies",
        "patterns": ["Do you have any hobbies?"],
        "responses": [
            "I enjoy helping people find information and answering questions!"
        ]
    },
    {
        "tag": "movie",
        "patterns": ["What's your favorite movie?"],
        "responses": [
            "I don't watch movies, but I've heard The Matrix is pretty interesting!"
        ]
    },
    {
        "tag": "game",
        "patterns": ["play a game", "Game", "play","hangman","20 questions","algebra","quiz"],
        "responses": [
            "Sure! How about a game of hangman, a game of questions, or an algebra problem?",
            "Great! Do you want to play algebra, hangman, or 20 questions?"
        ]
    },
    {
        "tag": "story",
        "patterns": ["Story", "generate a story?", "Create a fantasy story", "I want a mystery story"],
        "responses": [
            "Story ",
            "Sure! What genre of story would you like? Fantasy, mystery, or adventure?",
            "In a magical kingdom, there was a young wizard who...",
            
        ]
    },
    {
        "tag": "music",
        "patterns": ["Recommend me some music", "Music", "song", "Suggest a rock song", "I need some jazz music"],
        "responses": [
            "Sure! What genre do you like? Pop, rock, jazz, or classical?",
            "I can recommend some songs. What genre are you in the mood for?",
            "I can suggest some songs. What genre are you in the mood for?",
            "just talk me ,then i will recommend you music"
        ]
    },
    {
        "tag": "order",
        "patterns": ["Can you help me with my order?", "I need help with my account", "I forgot my username"],
        "responses": [
            "Sure, I can help with your order. Please provide your order number.",
            "I'd be happy to help with your account. What seems to be the problem?",
            "To recover your username, click on 'Forgot Username' on the login page and follow the instructions."
        ]
    },
    {
        "tag": "payment",
        "patterns": ["What payment methods do you accept?"],
        "responses": [
            "We accept credit cards, debit cards, and PayPal."
        ]
    },
    {
        "tag": "settings",
        "patterns": ["How do I change my account settings?"],
        "responses": [
            "To change your account settings, log in to your account and go to the settings page."
        ]
    },
    {
        "tag": "gift_cards",
        "patterns": ["Do you offer gift cards?"],
        "responses": [
            "Yes, we offer gift cards. You can purchase them on our website."
        ]
    }
    
]
}
    

def chatbot_response(user_input):
    """
    Generate a response from the chatbot based on user input.
    """
    # Transform user input
    x_test = loaded_vectorizer.transform([user_input])

    # Predict tag
    predicted_tag = loaded_model.predict(x_test)[0]

    # Find response
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "I'm sorry, I didn't understand that."

# Example usage
if __name__ == "__main__":
    print("Chatbot is ready. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
