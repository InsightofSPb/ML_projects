# Классификация фейковых новостей на основе Fake News датасете

Link: https://www.kaggle.com/c/fake-news/data

### Что применялось:
+ PorterStemmer
+ TfidfVectorizer
+ xgboost
+ PassiveAggressiveClassifier
+ BayesSearchCV

### Результаты:
+ Наилучший результат по F1-macro для XGBoost - 0.9673
+ Визуализация наиболее встречающихся слов в датасете на основании Tfidf

![top50](https://github.com/InsightofSPb/ML_projects/blob/main/Fake_news/Pictures/Top50words.png)
![models](https://github.com/InsightofSPb/ML_projects/blob/main/Fake_news/Pictures/Colorbar.png)

## Загрузка библиотек
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия
from sklearn.neighbors import KNeighborsClassifier  # Алгоритм ближайших соседей
from sklearn.tree import DecisionTreeClassifier  # Дерево решений
from sklearn.ensemble import RandomForestClassifier  # Алгоритм случайного леса
from nltk import stem
from nltk.corpus import stopwords
import re
import circlify
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real
import os
```

## Предобработка датасета
Удаляем пустые значения в колонке `text` и не будем обучать модель на основе заголовков и авторов
```python
df = pd.read_csv('train.csv')
df = df.dropna(subset=['text'])
df = df.drop(labels=['title', 'author'], axis=1)
```

## Стемминг
Убираем все не цифры и не слова, удаляем стоп-слова, проводим стемминг
```python
stemmer = stem.PorterStemmer()
# nltk.download('stopwords')
stopwords = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = [el for el in text.split() if el not in stopwords]
    text = " ".join([stemmer.stem(el) for el in text])
    return text


df['text'] = df['text'].apply(preprocess)
```


## Делим выборку и применяет векторизацию
```python
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.20, random_state=42)

tfidf = TfidfVectorizer(max_df=0.7)  # get rid of too frequent words (0.7)

tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)
```


## Обучаем разные методы классификации
```python
pac = PassiveAggressiveClassifier(max_iter=500, C=0.8)
svm = SVC(C=0.9)
knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine')
tree = DecisionTreeClassifier(max_depth=40)
rf = RandomForestClassifier(n_estimators=150, max_depth=50)
log = LogisticRegression(C=1.2)

estimators = [('clf', XGBClassifier(random_state=42, max_depth=80))]
pipe = Pipeline(steps=estimators)
search_space = {'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
                'clf__subsample': Real(0.5, 1.0),
                'clf__colsample_bytree': Real(0.5, 1.0),
                'clf__colsample_bylevel': Real(0.5, 1.0),
                'clf__colsample_bynode': Real(0.5, 1.0),
                'clf__reg_alpha': Real(0.0, 10.0),
                'clf__reg_lambda': Real(0.0, 10.0),
                'clf__gamma': Real(0.0, 10.0)}

opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=40, scoring='f1', random_state=42)

pac.fit(tfidf_train, y_train)
print('Агрессивный обучился!')
svm.fit(tfidf_train, y_train)
print('SVM обучился!')
knn.fit(tfidf_train, y_train)
print('KNN обучился!')
tree.fit(tfidf_train, y_train)
print('Дерево обучился!')
rf.fit(tfidf_train, y_train)
print('Лес обучился!')
log.fit(tfidf_train, y_train)
print('Логистическая обучилась!')
opt.fit(tfidf_train, y_train)
print('XGB обучился!')
```


## Предсказываем
```python
y_pred_pac = pac.predict(tfidf_test)
y_pred_svm = svm.predict(tfidf_test)
y_pred_knn = knn.predict(tfidf_test)
y_pred_tree = tree.predict(tfidf_test)
y_pred_rf = rf.predict(tfidf_test)
y_pred_log = log.predict(tfidf_test)
y_pred_xgb = opt.predict(tfidf_test)
```

## Выводим датасет с результатами
```python
target_names = ['REAL', 'FAKE']

y_preds = [y_pred_pac, y_pred_svm, y_pred_knn, y_pred_tree, y_pred_rf, y_pred_log]
precision = []
recall = []
f1 = []
for i in range(len(y_preds)):
    y = y_preds[i]
    precision.append(round(precision_recall_fscore_support(y_test, y, average='macro')[0], 4))
    recall.append(round(precision_recall_fscore_support(y_test, y, average='macro')[1], 4))
    f1.append(round(precision_recall_fscore_support(y_test, y, average='macro')[2], 4))

columns = ['PassArgClf', 'SVM', 'KNeigh', 'DesTree', 'RanForest', 'LogRegr']

D = {}

for i in range(len(columns)):
    D[columns[i]] = (float(precision[i]), float(recall[i]), float(f1[i]))
df = pd.DataFrame(D, index=['precision', 'recall', 'f1'])
df['XGB'] = ['-', '-', round(f1_score(y_test, y_pred_xgb), 4)]
cwd = os.getcwd()
path = cwd + "/res.csv"
df.to_csv(path, index=False)
print(df.head())
```

## Функция для вывода наилучшей матрицы смещения
```python
def bestconfmat(y=y_pred_xgb):
    cnfm = confusion_matrix(y_test, y, normalize='true')
    df = pd.DataFrame(cnfm, index=[el for el in target_names], columns=[el for el in target_names])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df, annot=True, fmt=".4f")
    plt.title(f'Матрица смещения для XGBoost')
    plt.savefig('ConfMatrix.png')
    plt.show()
```

## Функция для вывода 50 самых популярных слов
```python
def plotwords(n=50):
    mean_weights = np.asarray(tfidf_train.mean(axis=0)).ravel().tolist()

    mean_df = pd.DataFrame({'token': tfidf.get_feature_names_out(), 'mean_weight': mean_weights})
    N = 50

    mean_df = mean_df.sort_values(by='mean_weight', ascending=False).reset_index(drop=True)
    print(mean_df.head(N))

    circles = circlify.circlify(mean_df['mean_weight'][0:N].tolist(),
                                show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0))
    fig, ax = plt.subplots(figsize=(12,9), facecolor='white')
    ax.axis('off')
    lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r) for circle in circles)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    labels = list(mean_df['token'][0:N])
    counts = list(round(mean_df['mean_weight'][0:N] * 100, 2))
    labels.reverse()
    counts.reverse()
    cmap = plt.colormaps['cool']
    iters = 0
    for circle, label, count in zip(circles, labels, counts):
        x, y, r = circle
        color = cmap(iters/len(circles))
        ax.add_patch(plt.Circle((x, y), r, color=color))
        plt.annotate(label + '\n' + str(count), (x, y), size=12, va='center', ha='center')
        iters += 1
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Top {N} stems (words) in the dataset', size=16)
    plt.savefig('Top50words.png')
    plt.show()
```

## Сравнение обученных моделей между собой
```python
models = pd.DataFrame({
    "Model": ['PassiveAggressiveClassifier', 'SVM', 'KNeighbors', 'DecisionTree','RandomForest','LogisticRegression', 'XGBoost'],
    "Score": [pac.score(tfidf_test, y_test),
              svm.score(tfidf_test, y_test),
              knn.score(tfidf_test, y_test),
              tree.score(tfidf_test, y_test),
              rf.score(tfidf_test, y_test),
              log.score(tfidf_test, y_test),
              opt.score(tfidf_test, y_test)]
})

bestconfmat()
plotwords()

plt.figure(figsize=(15, 8))
sn.set_style('whitegrid')
sn.barplot(x=models['Model'], y=models['Score'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.yticks(np.arange(0.0, 1.0, 0.05))
plt.title("Model Selection")
plt.savefig('Colorbar.png')
plt.show()
```
