import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


stopwords = ["a", "the"]
data = pd.read_csv("spam.csv", encoding="latin-1")

messages = data["v2"]

message_types = data["v1"]

def remove_stopwords(messages):
    processed_messages = []
    for sentence in messages:
        list = [word for word in sentence.split(
            " ") if (word not in stopwords)]
        processed_sentence = " ".join(list)
        processed_messages.append(processed_sentence)
    return processed_messages


processed_messages = remove_stopwords(messages)

vectorizer = TfidfVectorizer()
vectorized_X = vectorizer.fit_transform(processed_messages)

message_train, message_test, message_types_train, message_types_test = train_test_split(vectorized_X, message_types, test_size=0.3, random_state=20) 

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, message_types_train)

# Test
dataForPredict = remove_stopwords(pd.DataFrame(data={
    "v2": ["Dear Voucher Holder, To claim this weeks offer you have to become our partner"]
})["v2"])

X_val = vectorizer.transform(dataForPredict)

result = Spam_model.predict(X_val)[0]
print(f"Your text is '{result}'")