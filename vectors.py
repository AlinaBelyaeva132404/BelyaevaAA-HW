import pandas as pd
import numpy as np
import re
import os
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB
from scipy.stats import uniform, norm
import warnings
warnings.filterwarnings('ignore')


# Папка для сохранения всех графиков
output_dir = r"C:/Users/alino/Downloads/plots"
os.makedirs(output_dir, exist_ok=True)


# Кастомный словарь стоп-слов
custom_stopwords = [
    'не', 'но', 'за', 'очень', 'так', 'на', 'х1', 'как', 'что',
    'заказ', 'они', 'ней', 'при', 'то', 'бы', 'буду', 'все', 'по', 'до', 'он', 'мне'
]


# Загрузка данных
df = pd.read_csv("C:/Users/alino/Downloads/Telegram Desktop/reviews.csv", sep="\t")


# Предобработка текста с кастомными стоп-словами
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in custom_stopwords]
    return " ".join(words)

df['clean_text'] = df['review'].apply(preprocess_text)


# CountVectorizer и TF-IDF
cv_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=custom_stopwords)
tf_cv = cv_vectorizer.fit_transform(df['clean_text'])
feature_names_cv = cv_vectorizer.get_feature_names_out()

tfidf_vectorizer = TfidfVectorizer(max_features=1000, max_df=0.5, stop_words=custom_stopwords, smooth_idf=True)
tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])
feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()


# Гистограммы частот слов
word_counts = np.array(tf_cv.sum(axis=0)).flatten()
plt.figure(figsize=(10,6))
plt.hist(word_counts, bins=50, color='skyblue')
plt.title("Распределение частот слов (CountVectorizer)")
plt.xlabel("Частота слова")
plt.ylabel("Количество слов")
plt.savefig(os.path.join(output_dir, "CountVectorizer_word_freq.png"))
plt.close()

tfidf_vals = tfidf.data
plt.figure(figsize=(10,6))
plt.hist(tfidf_vals, bins=50, color='lightgreen')
plt.title("Распределение TF-IDF значений")
plt.xlabel("TF-IDF")
plt.ylabel("Количество")
plt.savefig(os.path.join(output_dir, "TFIDF_distribution.png"))
plt.close()


# GaussianNB пример
df['weight'] = np.random.randint(50, 120, df.shape[0])
df['height'] = np.random.randint(150, 200, df.shape[0])
df['age'] = np.random.randint(18, 70, df.shape[0])
df['cardio'] = np.random.randint(0,2, df.shape[0])

gnb = GaussianNB()
train = df[['age', 'weight', 'height']]
target = df['cardio']
model = gnb.fit(train, target)
predict = model.predict(train)

prob = model.predict_proba(train)[:,1]
plt.figure(figsize=(10,6))
plt.hist(prob[target==1], bins=np.linspace(0,1,50), alpha=0.5, label='1')
plt.hist(prob[target==0], bins=np.linspace(0,1,50), alpha=0.5, label='0')
plt.axvline(0.5, color='black')
plt.legend()
plt.title("GaussianNB: вероятности классов")
plt.savefig(os.path.join(output_dir,"GaussianNB_probabilities.png"))
plt.close()


# Boxplot и гистограммы (weight)
plt.figure(figsize=(10,6))
plt.hist(df['weight'], bins=10, color='orange')
plt.title("Гистограмма веса")
plt.savefig(os.path.join(output_dir,"weight_hist.png"))
plt.close()

plt.figure(figsize=(10,6))
plt.boxplot(df['weight'])
plt.title("Boxplot веса")
plt.savefig(os.path.join(output_dir,"weight_boxplot.png"))
plt.close()


# Распределения Uniform и Normal
x = np.linspace(-0.2,1.2,100)
plt.figure(figsize=(8,5))
plt.plot(x, uniform.pdf(x))
plt.title("Uniform PDF")
plt.savefig(os.path.join(output_dir,"uniform_pdf.png"))
plt.close()
plt.figure(figsize=(8,5))
plt.plot(x, uniform.ppf(x))
plt.title("Uniform PPF")
plt.savefig(os.path.join(output_dir,"uniform_ppf.png"))
plt.close()

x = np.random.uniform(0,1,10000)
plt.figure(figsize=(10,6))
plt.hist(x, density=True, bins=50, cumulative=True, alpha=0.6)
plt.plot(np.linspace(0,1,100), uniform.cdf(np.linspace(0,1,100)), 'r')
plt.title("Cumulative Uniform")
plt.savefig(os.path.join(output_dir,"uniform_cumulative.png"))
plt.close()

x = np.random.normal(0,1,10000)
plt.figure(figsize=(10,6))
plt.hist(x, density=True, bins=50, alpha=0.6)
plt.plot(np.linspace(x.min(), x.max(),100), norm.pdf(np.linspace(x.min(), x.max(),100)),'r')
plt.title("Normal distribution PDF")
plt.savefig(os.path.join(output_dir,"normal_distribution.png"))
plt.close()

x = np.random.normal(0,1,100)
plt.figure(figsize=(10,6))
plt.hist(x, density=True, bins=50, cumulative=True, alpha=0.6)
plt.plot(np.linspace(x.min(), x.max(),100), norm.cdf(np.linspace(x.min(), x.max(),100)),'r')
plt.title("Normal distribution CDF")
plt.savefig(os.path.join(output_dir,"normal_cumulative.png"))
plt.close()


# LSA (SVD) и WordCloud
n_topics_svd = 5
svd_model = TruncatedSVD(n_components=n_topics_svd, n_iter=100, random_state=42)
svd_model.fit(tfidf)

for topic_idx, topic in enumerate(svd_model.components_):
    word_freq = {feature_names_tfidf[i]: topic[i] for i in topic.argsort()[:-16:-1]}
    wc = WordCloud(width=800, height=600, background_color='white', colormap='tab10').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10,7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"LSA Topic #{topic_idx+1}")
    plt.savefig(os.path.join(output_dir, f"LSA_Topic_{topic_idx+1}.png"))
    plt.close()


# LDA и WordCloud
n_topics_lda = 5
lda = LatentDirichletAllocation(n_components=n_topics_lda, max_iter=15,
                                learning_method='online', learning_offset=50., random_state=42)
lda.fit(tf_cv)

for topic_idx, topic in enumerate(lda.components_):
    word_freq = {feature_names_cv[i]: topic[i] for i in topic.argsort()[:-16:-1]}
    wc = WordCloud(width=800, height=600, background_color='white', colormap='tab10').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10,7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"LDA Topic #{topic_idx+1}")
    plt.savefig(os.path.join(output_dir, f"LDA_Topic_{topic_idx+1}.png"))
    plt.close()
    
    
# Распределение доминирующих тем LDA
topic_distribution = lda.transform(tf_cv)
dominant_topics = np.argmax(topic_distribution, axis=1)
plt.figure(figsize=(8,5))
sns.countplot(x=dominant_topics, palette='pastel')
plt.title("Распределение доминирующих тем LDA")
plt.xlabel("Тема")
plt.ylabel("Количество документов")
plt.savefig(os.path.join(output_dir,"LDA_dominant_topics.png"))
plt.close()


# Расчёт P(w|d) для всех слов и документов
phi = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]  # normalize per topic
theta = topic_distribution
theta_T = theta.T

n_docs = tf_cv.shape[0]
n_words = len(feature_names_cv)
P_w_d = np.zeros((n_docs, n_words), dtype=np.float32)
for i in range(n_words):
    P_w_d[:, i] = phi[:, i] @ theta_T

for i in range(n_words):
    P_w_d[:, i] = phi[:, i] @ theta_T  # matrix multiplication

P_w_d_df = pd.DataFrame(P_w_d, columns=feature_names_cv)
P_w_d_df.to_csv(r"C:/Users/alino/Downloads/P_w_d.csv", index=False)

print(f"Все графики сохранены в папке: {output_dir}")
print("P(w|d) для всех слов и документов сохранено в CSV.")
