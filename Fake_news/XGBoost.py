import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import stem
from nltk.corpus import stopwords
import re
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real


df = pd.read_csv('train.csv')
df = df.dropna(subset=['text'])
df = df.drop(labels=['title', 'author'], axis=1)
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

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.20, random_state=42)


estimators = [('tfidf', TfidfVectorizer(max_df=0.7)),
              ('clf', XGBClassifier(random_state=42, max_depth=70))]
pipe = Pipeline(steps=estimators)


search_space = {'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
                'clf__subsample': Real(0.5, 1.0),
                'clf__colsample_bytree': Real(0.5, 1.0),
                'clf__colsample_bylevel': Real(0.5, 1.0),
                'clf__colsample_bynode': Real(0.5, 1.0),
                'clf__reg_alpha': Real(0.0, 10.0),
                'clf__reg_lambda': Real(0.0, 10.0),
                'clf__gamma': Real(0.0, 10.0)}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=1, scoring='f1', random_state=42)

opt.fit(X_train, y_train)

print(f'Лучшие параметры: {opt.best_estimator_}')
print(f'Для тренировочных данных f1-мера:  {opt.best_score_}')
print(f'Для тестовых данных f1-мера: {opt.score(X_test, y_test)}')
