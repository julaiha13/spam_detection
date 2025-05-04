from django.shortcuts import render

# Create your views here.
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from .forms import MessageForm




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'emails.csv')

dataset = pd.read_csv(csv_path)



vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dataset['text'])

x_train, x_test, y_train, y_test = train_test_split(x, dataset['spam'], test_size=0.2)

model = MultinomialNB()
model.fit(x_train, y_train,)

def predictMessage(message):
  messageVector = vectorizer.transform([message])
  prediction = model.predict(messageVector)
  return 'spam' if prediction[0] == 1 else 'Ham'

def Home(request):
    result = None
    if request.method == 'POST':
      form = MessageForm(request.POST)
      if form.is_valid():
        message = form.cleaned_data['text']
        result = predictMessage(message)
    else:
       form = MessageForm()

    return render(request, 'home.html', {'form': form, 'result':result})


