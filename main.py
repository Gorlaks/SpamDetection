import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

stopwords = ["a", "the"]
data = pd.read_csv("data.csv", encoding="utf-8")

text = data["text"]
X = []

y = data["type"]

labels = {
    "0": "spam",
    "1": "normal",
    "2": "crap"
}


def remove_stopwords(text, X):
    for sentence in text:
        list = [word for word in sentence.split(
            " ") if (word not in stopwords)]
        processed_sentence = " ".join(list)
        X.append(processed_sentence)
    return X


X = remove_stopwords(text, X)

vectorizer = TfidfVectorizer()
digital_X = vectorizer.fit_transform(X)

model = DecisionTreeClassifier(random_state=0)
model.fit(digital_X.toarray(), y)


dataForPredict = pd.DataFrame(data={
    "text": ["Привет. Ты сделал свою работу?"]
})["text"]

X_val = vectorizer.transform(dataForPredict)

result = str(model.predict(X_val.toarray())[0])
print(f"Your text is '{labels[result]}'")