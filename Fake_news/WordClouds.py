import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import stem
from nltk.corpus import stopwords
import re
import circlify
import matplotlib.pyplot as plt


df = pd.read_csv('news.csv')
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

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)  # get rid of stop-words and too frequent words (0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)


def plotwords(n=50):
    mean_weights = np.asarray(tfidf_train.mean(axis=0)).ravel().tolist()

    mean_df = pd.DataFrame({'token': tfidf.get_feature_names_out(), 'mean_weight': mean_weights})
    N = n

    mean_df = mean_df.sort_values(by='mean_weight', ascending=False).reset_index(drop=True)
    print(mean_df.head(N))

    circles = circlify.circlify(mean_df['mean_weight'][0:N].tolist(),
                                show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0))
    n = mean_df['mean_weight'][0:N].max()
    fig, ax = plt.subplots(figsize=(12,9), facecolor='white')
    ax.axis('off')
    lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r) for circle in circles)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    labels = list(mean_df['token'][0:N])
    counts = list(round(mean_df['mean_weight'][0:N] * 100, 2))
    labels.reverse()
    counts.reverse()
    cmap = plt.cm.get_cmap('cool')
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
    plt.show()
    plt.savefig('Top50words.png')

plotwords()