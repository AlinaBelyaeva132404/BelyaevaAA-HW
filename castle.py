import re
import numpy as np
import pandas as pd
from nltk.tokenize import regexp_tokenize
from pymorphy3 import MorphAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# семантические словари
FORTRESS_WORDS = {"крепость", "стена", "башня", "холм", "каменный", "древний", "город"}
LOCK_WORDS = {"дверь", "ключ", "открыть", "висеть", "запереть", "замочная"}

def extract_label_and_clean(text):
    if "за`мок" in text:
        label = "за`мок"
    elif "замо`к" in text:
        label = "замо`к"
    else:
        return None, None

    clean = text.replace("`", "").lower().strip()
    return clean, label


rows = []
with open(r"C:\Users\alino\Downloads\Telegram Desktop\замок.test",
          encoding="utf-8") as f:
    for line in f:
        t = line.strip()
        if not t:
            continue
        clean, label = extract_label_and_clean(t)
        if label:
            rows.append([clean, label])

df = pd.DataFrame(rows, columns=["text", "label"])

morph = MorphAnalyzer()

def tokenize_lemmas(text):
    return [
        morph.parse(tok)[0].normal_form
        for tok in regexp_tokenize(text, r"(?u)\b\w+\b")
    ]


def extract_context(text, window=8):
    tokens = tokenize_lemmas(text)
    if "замок" not in tokens:
        return None
    i = tokens.index("замок")
    left = tokens[max(0, i - window):i]
    right = tokens[i + 1:i + 1 + window]
    return " ".join(left + right)


def semantic_hint(context):
    tokens = context.split()
    f = any(w in tokens for w in FORTRESS_WORDS)
    l = any(w in tokens for w in LOCK_WORDS)
    return [int(f), int(l)]


df["context"] = df["text"].apply(extract_context)
df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(
    df["context"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=1
)

Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# создаём бинарные семантические признаки
Xtr_sem = np.array([semantic_hint(c) for c in X_train])
Xte_sem = np.array([semantic_hint(c) for c in X_test])

# объединяем TF-IDF и семантические признаки
Xtr_final = hstack([Xtr, Xtr_sem])
Xte_final = hstack([Xte, Xte_sem])

clf = LogisticRegression(
    max_iter=500,
    class_weight="balanced"
)

clf.fit(Xtr_final, y_train)

preds = clf.predict(Xte_final)

print(classification_report(y_test, preds))


def predict_text(text):
    ctx = extract_context(text.lower())
    if ctx is None:
        return "В тексте нет слова 'замок'"

    vec_tfidf = tfidf.transform([ctx])
    vec_sem = np.array([semantic_hint(ctx)])
    vec_final = hstack([vec_tfidf, vec_sem])

    pred = clf.predict(vec_final)[0]
    probs = clf.predict_proba(vec_final)[0]

    return pred, {
        clf.classes_[0]: probs[0],
        clf.classes_[1]: probs[1]
    }


input_text = "я ввзял в руки замок"

prediction, scores = predict_text(input_text)

print("ПРЕДСКАЗАНИЕ ДЛЯ ПОЛЬЗОВАТЕЛЬСКОГО ТЕКСТА")
print("Текст:", input_text)
print("Результат:", prediction)
print("Вероятности:")
for k, v in scores.items():
    print(f"  {k}: {v:.4f}")
