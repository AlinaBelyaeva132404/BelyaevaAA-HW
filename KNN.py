import re
import contractions
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from gensim.models import FastText
import matplotlib.pyplot as plt
from wordcloud import WordCloud

tqdm.pandas()  # для прогресса при apply

# 1. Загрузка
df = pd.read_csv("C:/Users/alino/Downloads/Telegram Desktop/reviews.csv", sep="\t")
if 'review' not in df.columns:
    raise ValueError("Ожидается столбец 'review' в CSV.")

# 2. Предобработка с прогресс-баром
stop_ru = [
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
    "его","но","они","еще","бы","за","вы","мы","этот","была","быть","чтобы","только",
    "ее","или","у","же","по","из","к","от","до","для","про","ну","там","тут","очень"
]
stop_en = []
stop_words = list(set(stop_ru).union(stop_en))

def clean_text(text):
    if pd.isna(text):
        text = ""
    text = contractions.fix(str(text))
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

print("Предобработка текста...")
df['clean'] = df['review'].progress_apply(clean_text)

# 3. Векторизация
print("\nВекторизация TF-IDF...")
vectorizer_tfidf = TfidfVectorizer(
    token_pattern=r'\b[а-яёa-z]{3,}\b',
    stop_words=stop_words,
    min_df=5,
    max_df=0.95,
    ngram_range=(1,2)
)
X_tfidf = vectorizer_tfidf.fit_transform(df['clean'])

print("\nВекторизация BoW...")
vectorizer_bow = CountVectorizer(
    token_pattern=r'\b[а-яёa-z]{3,}\b',
    stop_words=stop_words,
    min_df=0.001,
    max_df=0.99,
    ngram_range=(1,2)
)
X_bow = vectorizer_bow.fit_transform(df['clean'])

# 4. Снижение размерности
use_svd = True
n_components = 100
if use_svd:
    print("\nПрименение SVD...")
    svd = TruncatedSVD(n_components=min(n_components, X_tfidf.shape[1]-1), random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_reduced = lsa.fit_transform(X_tfidf)
else:
    X_reduced = X_tfidf

# 5. KMeans на TF-IDF
k_final = 4
print(f"\nОбучение KMeans с {k_final} кластерами...")
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=20, max_iter=10000)
cluster_labels = kmeans.fit_predict(X_reduced)  # без tqdm
df['cluster'] = cluster_labels

# 6. Топ-слова для интерпретации
if use_svd:
    try:
        comp = lsa.named_steps['truncatedsvd'].components_
        centers_in_tfidf = np.dot(kmeans.cluster_centers_, comp)
    except:
        centers_in_tfidf = np.zeros((k_final, X_tfidf.shape[1]))
        for i in range(k_final):
            idx = np.where(cluster_labels == i)[0]
            if len(idx) > 0:
                centers_in_tfidf[i,:] = X_tfidf[idx].mean(axis=0)
else:
    centers_in_tfidf = kmeans.cluster_centers_

n_top_words = 15
terms_tfidf = vectorizer_tfidf.get_feature_names_out()
cluster_top_words = {}
for i in range(k_final):
    row = np.array(centers_in_tfidf[i]).ravel()
    top_idx = row.argsort()[-n_top_words:][::-1]
    cluster_top_words[i] = [terms_tfidf[j] for j in top_idx]

# 7. FastText
tokenized_docs = [doc.split() for doc in df['clean']]
print("\nОбучение FastText...")
ft_model = FastText(vector_size=300, window=5, min_count=2, workers=4, sg=1)
ft_model.build_vocab(corpus_iterable=tqdm(tokenized_docs, desc="FastText vocab"))
ft_model.train(
    corpus_iterable=tqdm(tokenized_docs, desc="FastText train"),
    total_examples=len(tokenized_docs),
    epochs=5
)

def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocab = set(model.wv.index_to_key)
    features = []
    for words in tqdm(corpus, desc="FT doc vectors"):
        vec = np.zeros(num_features)
        n = 0
        for w in words:
            if w in vocab:
                vec += model.wv[w]
                n += 1
        if n:
            vec /= n
        features.append(vec)
    return np.array(features)

doc_vecs_ft = averaged_word2vec_vectorizer(tokenized_docs, ft_model, 300)

# 8. KMeans на FastText векторизации
print("\nKMeans по FastText векторизации...")
km_ft = KMeans(n_clusters=k_final, random_state=42, n_init=20, max_iter=10000)
km_ft.fit(doc_vecs_ft)  # без tqdm
df['kmeans_cluster_ft'] = km_ft.labels_

# 9. Иерархическая кластеризация
def ward_hierarchical_clustering(feature_matrix):
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    return ward(cosine_distance)

sample_idx = np.random.choice(len(doc_vecs_ft), size=min(2000, len(doc_vecs_ft)), replace=False)
sample_vecs = doc_vecs_ft[sample_idx]
print("\nИерархическая кластеризация (Ward)...")
linkage_matrix = ward_hierarchical_clustering(sample_vecs)

plt.figure(figsize=(12,18))
dendrogram(linkage_matrix, truncate_mode='lastp', orientation="left", p=50)
plt.tight_layout()
plt.savefig('hierarchical_clusters.png', dpi=200)
plt.close()

# После KMeans и FastText кластеризации добавляем сохранение

# 1. Сохраняем DataFrame с кластерами
df_to_save = df[['review','clean','cluster','kmeans_cluster_ft']]
df_to_save.to_csv('reviews_clusters.csv', index=False, encoding='utf-8-sig')
print("Сохранён CSV с результатами кластеризации: reviews_clusters.csv")

# 2. Сохраняем топ-слов каждого кластера в CSV
import csv
with open('cluster_top_words.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['cluster', 'top_words'])
    for cl, words in cluster_top_words.items():
        writer.writerow([cl, ', '.join(words)])
print("Сохранён CSV с топ-словами по кластерам: cluster_top_words.csv")

# 3. Генерация WordCloud для каждого кластера
for cl, words in cluster_top_words.items():
    freq_dict = {w: 1 for w in words}
    wc = WordCloud(
        width=900,
        height=600,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    fname = f'wordcloud_cluster_{cl}.png'
    plt.savefig(fname, dpi=150)
    plt.close()

print("Готово. Файлы CSV и графики кластеров созданы.")
    print(f"Сохранён WordCloud для кластера {cl}: {fname}")
    plt.axis("off")

