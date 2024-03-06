import nltk
from nltk.chat.util import Chat, reflections
from fuzzywuzzy import process

# Define patterns for the chatbot
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you?', ['I am doing well, thank you!', 'I am fine, thanks for asking.']),
    (r'who are you?', ['I am Nexus-Bot, Your personal healthcare consultant!.']),
    (r'what do you do?', ['Providing you with a good information about Wellness-Nexus services.']),
    (r'quit|exit', ['Goodbye!', 'Bye!', 'See you later!']),
    (r'what services do you offer?', ['We offer a variety of services including Nexus-Scan, Nexus-Sugar, and Nexus-Meal.']),
    (r'what is Nexus-Scan?', ['Nexus-Scan, our Computer Vision model, works wonders with just a picture of your meal. It accurately determines the calorie content, helping you keep track of what you eat. By storing this data, it provides valuable insights into your long-term calorie intake, which is incredibly helpful in managing conditions like diabetes and obesity. With Nexus-Scan, you will receive personalized tips and suggestions to make healthier food choices and improve your overall well-being.']),
    (r'what is Nexus-Sugar?', ['Nexus-Sugar helps determine if your blood sugar levels are normal or related to diabetes.']),
    (r'what is Nexus-Fit?', ['Nexus-Meal provides information about your body condition based on variable like height, weight, age and gender. it helps you to stay tracked to normal weight']),
    (r'how does Nexus-Scan work?', ['Nexus-Scan uses advanced algorithms to analyze meal photos and estimate calorie content.']),
    (r'how does Nexus-Sugar work?', ['Nexus-Sugar analyzes your blood sugar data to detect patterns and trends over time.']),
    (r'how does Nexus-Meal work?', ['Nexus-Meal offers personalized advice and recommendations based on your dietary preferences.']),
]

# Add more patterns based on the provided information
patterns += [
    (r'CV to Diagnose food calories content and fat', ['Nexus-Scan is a computer vision model that diagnoses food calories content and fat by analyzing meal photos.']),
    (r'Reminder Algorithm', ['NexusBot uses a reminder algorithm to help users stay on track with their health goals and provides recommendations for delicious, healthy food options.']),
    (r'Sugar Blood Content measurement', ['Nexus-Sugar measures sugar blood content over time to analyze the probability of being healthy or diabetical.']),
    (r'Obesity Calculator', ['Nexus-Calculator calculates obesity based on personalized data and provides insights into health and fitness.']),
    (r'Website to provide all of it', ['Nexus-Web is a website that integrates all Wellness-Nexus services and provides comprehensive health information and resources.']),
    (r'Interface to make the apps looks healthy and intriguing', ['Nexus-FrontEnd designs interfaces that are visually appealing and engaging to encourage user interaction and adoption of healthy habits.']),
    (r'Good morning|Good morning!', ['Good morning!', 'Morning!', 'Hello, how are you today?']),
    (r'Good afternoon|Good afternoon!', ['Good afternoon!', 'Afternoon!', 'How can I assist you this afternoon?']),
    (r'Good evening|Good evening!', ['Good evening!', 'Evening!', 'How can I help you tonight?']),
    (r'How was your day?', ['It was good, thank you for asking.', 'Pretty good, thanks!', 'Not too bad, how about yours?']),
    (r'What are your plans for today?', ['Just here to assist you!', 'Helping users like you!', 'Answering your questions!']),
    (r'What do you like to do in your free time?', ['I enjoy chatting with users like you!', 'Learning new things!', 'Helping out wherever I can!']),

    # Jokes
    (r'Tell me a joke', ['Why dont scientists trust atoms? Because they make up everything!', 'I told my wife she should embrace her mistakes. She gave me a hug.', 'Why did the scarecrow win an award? Because he was outstanding in his field!']),
    (r'Can you make me laugh?', ['Sure, here\'s one: Why did the bicycle fall over? Because it was two-tired!', 'How do you organize a space party? You planet!', 'What do you call fake spaghetti? An impasta!']),
    (r'Knock knock', ['Who\'s there?', 'Knock knock!', 'Who\'s there?', 'Boo', 'Boo who?', 'Don\'t cry, it\'s just a joke!']),
    (r'Why did the chicken cross the road?', ['To get to the other side!', 'To prove it wasn\'t chicken!', 'To avoid Colonel Sanders!']),
    (r'What do you call a fish with no eyes?', ['Fsh!', 'A fish!', 'Blind!']),
    (r'Why was the math book sad?', ['Because it had too many problems!', 'Because it had too many chapters!', 'Because it had too many pages!']),
    (r'Why did the tomato turn red?', ['Because it saw the salad dressing!', 'Because it was embarrassed!', 'Because it was blushing!']),
    (r'What do you call cheese that isn\'t yours?', ['Nacho cheese!', 'Not your cheese!', 'Stolen cheese!']),
    (r'Why don\'t skeletons fight each other?', ['They don\'t have the guts!', 'They don\'t have the stomach for it!', 'They don\'t have the heart!']),
    (r'Why did the golfer bring two pairs of pants?', ['In case he got a hole in one!', 'In case he got a hole in three!', 'In case he got a hole in five!']),
    (r'How\'s the weather today?', ['Im not sure, but you can check your local weather forecast!', 'I dont have access to real-time weather information, but you can check online or on your phone.']),
    (r'What time is it?', ['I dont have access to real-time clock information, but you can check the time on your device!', 'You can check the time on your phone or computer.']),
    (r'Have you had breakfast/lunch/dinner?', ['I don\'t eat, but I\'m always here to help!', 'I don\'t require food, but I\'m ready to assist you!', 'I don\'t have meals like humans, but I\'m available to chat!']),
    (r'How\'s your day going so far?', ['Every day is a good day to assist users like you!', 'Im here and ready to help, so its going well!', 'I don\'t have days like humans, but I\'m always available to chat!']),

    # Jokes (continued)
    (r'Why dont scientists trust atoms?', ['Because they make up everything!', 'Because theyre always up to something!', 'Because they can\'t be seen with the naked eye!']),
    (r'What do you call fake spaghetti?', ['An impasta!', 'A faketti!', 'An imposter!']),
    (r'Why did the scarecrow win an award?', ['Because he was outstanding in his field!', 'Because he was so good at his job!', 'Because he was so scary!']),
    (r'What do you get when you cross a snowman and a vampire?', ['Frostbite!', 'Frosty the Snow Vampire!', 'A snowman that sucks your blood!']),
    (r'Why dont skeletons fight each other?', ['They dont have the guts!', 'They dont have the stomach for it!', 'They dont have the heart!']),
    (r'What did one hat say to the other?', ['You stay here, Ill go on ahead!', 'Youre on top of things!', 'Youre a head of the game!']),
    (r'Why did the bicycle fall over?', ['Because it was two-tired!', 'Because it was leaning too much!', 'Because it lost its balance!']),
    (r'What did one ocean say to the other ocean?', ['Nothing, they just waved!', 'They just went with the flow!', 'They just said "sea" you later!']),
    (r'Why did the tomato turn red?', ['Because it saw the salad dressing!', 'Because it was embarrassed!', 'Because it was blushing!']),
    (r'Why dont eggs tell jokes?', ['Because they might crack up!', 'Because theyre too chicken!', 'Because theyre not yolking around!']),

    # Random Interactions
    (r'Tell me something interesting', ['Did you know that the Earths atmosphere is about 78% nitrogen, 21% oxygen, and 1% other gases?', 'The longest recorded flight of a chicken is 13 seconds!', 'Cows have best friends and get stressed when they are separated!']),
    (r'Can you recommend a movie?', ['What genre do you like?', 'Sure, what type of movie are you in the mood for?', 'I can suggest a movie based on your favorite genre!']),
    (r'What\'s your favorite book?', ['I don\'t have preferences like humans, but I can recommend some popular books!', 'I enjoy all kinds of books, but I can suggest a few if you like.', 'I don\'t read books, but I can help you find something interesting to read!']),
    (r'Do you like music?', ['I don\'t have personal preferences, but I can help you find music recommendations!', 'I don\'t listen to music, but I can assist you in finding some good tunes!', 'I can help you discover new music if you\'d like!']),
    (r'What\'s your favorite color?', ['I don\'t have preferences like humans, but I can appreciate all colors!', 'I don\'t see colors, but I can help you find color inspiration if you need!', 'I don\'t have a favorite color, but I can help you pick one!']),
    (r'Can you tell me a fun fact?', ['Sure! Did you know that butterflies taste with their feet?', 'Here\'s one: The shortest war in history lasted only 38 minutes!', 'Did you know that an octopus has three hearts and blue blood?']),
    (r'Can you help me with a riddle?', ['Of course! I love riddles! Go ahead and ask.', 'Sure! Fire away with your riddle.', 'I\'d be happy to try! What\'s your riddle?']),
    (r'Are you a robot?', ['I am an AI-powered chatbot designed to assist users like you!', 'Yes, I am an AI-based virtual assistant here to help you!', 'I am an artificial intelligence designed to provide assistance and answer your questions!']),
]

# Create a chatbot instance
chatbot = Chat(patterns, reflections)

import pickle

# Save the chatbot model
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(chatbot, f)


# Start the conversation loop
print("Welcome to NexusBot! How can I assist you today?")
while True:
    user_input = input("You: ")

    # Perform fuzzy matching to find the best match for user input
    best_match, _ = process.extractOne(user_input, [pattern[0] for pattern in patterns])
    response = chatbot.respond(best_match)

    print("NexusBot:", response)
    if user_input.lower() == 'quit' or user_input.lower() == 'exit':
        break