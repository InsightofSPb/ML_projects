import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier
from nltk import stem
from nltk.corpus import stopwords
import re

df = pd.read_csv('news.csv')
# print(f'Size of data frame: {df.shape}')  # 6335 , 4
labels = df['label']
# print(f'Size of data frame: {df.shape}')  # 6335 , 4
stemmer = stem.SnowballStemmer('english')
# nltk.download('stopwords')
stopwords = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = [el for el in text.split() if el not in stopwords]
    text = " ".join([stemmer.stem(el) for el in text])
    return text


df['text'] = df['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.20, random_state=42)

tfidf = TfidfVectorizer(max_df=0.7)  # get rid of too frequent words (0.7)

tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

gbc = GradientBoostingClassifier(learning_rate=0.5, n_estimators=500, max_depth=5)
gbc.fit(tfidf_train, y_train)

y_pred_gbc = gbc.predict(tfidf_test)

print(precision_recall_fscore_support(y_test, y_pred_gbc, average='macro'))