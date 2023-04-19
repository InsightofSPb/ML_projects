import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from nltk import stem
from nltk.corpus import stopwords
import re
import nltk


df = pd.read_csv('news.csv')
labels = df['label']

stemmer = stem.SnowballStemmer('english')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = [el for el in text.split() if el not in stopwords]
    text = " ".join([stemmer.stem(el) for el in text])
    return text


df['text'] = df['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.20, random_state=42)

tfidf = TfidfVectorizer(max_df=0.7)  # get rid of stop-words and too frequent words (0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

tree_final_model=DecisionTreeClassifier(max_depth=58,random_state=42)
tree_final_model.fit(tfidf_train,y_train)
tree_training_acc = tree_final_model.score(tfidf_train,y_train)
tree_testing_acc = tree_final_model.score(tfidf_test,y_test)
print(f"Training accuracy of DesicionTreeClassifier is {tree_training_acc}")
print(f"testing accuracy of DesicionTreeClassifier is {tree_testing_acc}")
