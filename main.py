import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

stopwords = ["a", "the"]
data = pd.read_csv("spam.csv", encoding="latin-1")

v2 = data["v2"]
X = []

y = data["v1"]

labels = {
    "0": "ham",
    "1": "spam"
}


def remove_stopwords(v2, X):
    for sentence in v2:
        list = [word for word in sentence.split(
            " ") if (word not in stopwords)]
        processed_sentence = " ".join(list)
        X.append(processed_sentence)
    return X


X = remove_stopwords(v2, X)

vectorizer = TfidfVectorizer()
digital_X = vectorizer.fit_transform(X)

model = DecisionTreeClassifier(random_state=0)
model.fit(digital_X.toarray(), y)


dataForPredict = pd.DataFrame(data={
    "v2": ["Ok lar..."]
})["v2"]

X_val = vectorizer.transform(dataForPredict)

result = str(model.predict(X_val.toarray())[0])
print(labels[result])