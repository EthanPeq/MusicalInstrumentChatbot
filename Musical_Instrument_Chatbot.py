import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

questions = []
answers = []

with open('qa_Musical_Instruments.json','r') as f:
    for line in f:
        data = ast.literal_eval(line)
        questions.append(data['question'].lower())
        answers.append(data['answer'].lower())
        
print('Data on questions:', len(questions))

##for index, val in enumerate(questions[20]):
##    print('[', index, ']', val)

# Use CountVectorizer to convert the questions list into a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer(stop_words = 'english')
X_vec = vectorizer.fit_transform(questions)
tfidf = TfidfTransformer(norm='l2')
X_tfidf = tfidf.fit_transform(X_vec)

def conversation(quest):
    global tfidf, answers, X_tfidf
    #quest is the users question the we (the chatbot) is to answer
    Y_vect = vectorizer.transform(quest)
    Y_tfidf = tfidf.fit_transform(Y_vect)

    angle = np.rad2deg(np.arccos(max(cosine_similarity(Y_tfidf, X_tfidf)[0])))
    if angle > 60:
        return "Sorry, I did not quite understand that"
    else:
        return answers[np.argmax(cosine_similarity(Y_tfidf,X_tfidf)[0])]

user = input('Please enter your name:')
print('Hi ', user, "I am a rep of Q&A support. How can I help you?")

while True:
    userquestion = input("{}: ".format(user))
    if userquestion.lower() == 'bye':
        print('Q&A: support: bye! Have a good day G')
        break
    else:
        print('The question is ', userquestion)
        print('Q&A support: ', conversation([userquestion]))
        
    

    
