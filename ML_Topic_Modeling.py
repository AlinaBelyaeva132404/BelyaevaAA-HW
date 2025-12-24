import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from wordcloud import WordCloud

from tqdm import tqdm

# лемматизация для русского языка
from pymorphy3 import MorphAnalyzer
morph = MorphAnalyzer()

tqdm.pandas()


# Папка для сохранения всех графиков
output_dir = r"C:/Users/alino/Downloads/plots_ML"
os.makedirs(output_dir, exist_ok=True)


# 1. Загрузка датасета
df = pd.read_csv("C:/Users/alino/Downloads/Telegram Desktop/reviews.csv", sep="\t")


# 2. Предобработка текста
def clean_text(text): # Нижний регистр, оставляем только кириллицу и цифры, убираем лишние пробелы
    text = str(text).lower()
    text = re.sub(r'[^а-яё0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean"] = df["review"].progress_apply(clean_text)


# лемматизация
def lemmatize_text(text):
    words = text.split()
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)

df["clean_lemma"] = df["clean"].progress_apply(lemmatize_text)


# Стоп-слова
russian_stopwords = [
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
    "его","но","они","еще","бы","за","вы","мы","этот","была","быть","чтобы","только",
    "ее","или","у","же","по","из","к","от","до","для","про","ну","там","тут","очень",
    "это", "итоге", "даже", "вот", "ещё", "этом", "мне", "вообще", "того", "всё", "при"
]


# 3. Векторизация текста
vectorizer = CountVectorizer(
    token_pattern=r"\b[а-яё]{3,}\b",
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2),
    stop_words=russian_stopwords
)

X = vectorizer.fit_transform(df["clean_lemma"])
terms = vectorizer.get_feature_names_out()


# 4. Обучение LDA
n_topics = 5
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method="batch",
    max_iter=20,
    verbose=1
)

print("Обучение LDA")
lda.fit(X)


# оценка качества модели
print("Perplexity:", lda.perplexity(X))


# 5. Вывод тем
def print_topics(model, terms, n_top_words=12):
    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[-n_top_words:][::-1]
        top_words = [terms[i] for i in top_indices]
        print(f"Тема {topic_idx}: {', '.join(top_words)}")

print("Темы, найденные LDA")
print_topics(lda, terms)


# 6. Присвоение темы каждому документу
doc_topic_dist = lda.transform(X)
df["dominant_topic"] = doc_topic_dist.argmax(axis=1)
df["topic_strength"] = doc_topic_dist.max(axis=1)


# распределение тем
topic_counts = df["dominant_topic"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(x=topic_counts.index, y=topic_counts.values)
plt.xlabel("Тема")
plt.ylabel("Количество документов")
plt.title("Распределение тем в корпусе отзывов")

plt.savefig(os.path.join(output_dir, "topic_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()


# примеры отзывов по темам
for topic in range(n_topics):
    print(f"\n{topic}")
    examples = df[df["dominant_topic"] == topic]["review"].head(3)
    for i, text in enumerate(examples, 1):
        print(f"{i}. {text}")


# 7. WORDCLOUD
def plot_wordcloud_for_topic(topic_idx, components, terms, top_n=50):
    topic = components[topic_idx]
    top_idx = topic.argsort()[-top_n:]
    top_words = {terms[i]: topic[i] for i in top_idx}

    wc = WordCloud(width=900, height=600, background_color="white")
    wc.generate_from_frequencies(top_words)

    plt.figure(figsize=(10, 6))
    plt.title(f"WordCloud для темы {topic_idx}", fontsize=16)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    plt.savefig(
        os.path.join(output_dir, f"wordcloud_topic_{topic_idx}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


for i in tqdm(range(n_topics), desc="WordCloud", unit="тема"):
    plot_wordcloud_for_topic(i, lda.components_, terms)


# 8. Сохранение результатов
csv_path = os.path.join(output_dir, "reviews_with_topics_ML.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Файл сохранён: {csv_path}")
