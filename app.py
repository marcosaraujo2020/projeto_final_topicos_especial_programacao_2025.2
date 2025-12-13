# Streamlit app: Classificação Binária de Textos com LIME e SHAP
# Projeto para disciplina "Tópicos Especiais em Programação"
# Dataset: SMS Spam Collection
# Single-file app with "pages" simulated via sidebar for 5 integrantes
# Requisitos (install):
# pip install streamlit scikit-learn pandas matplotlib seaborn lime shap joblib

import os
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Interpretable libraries
from lime.lime_text import LimeTextExplainer
import shap

# ----------------------- Utilities -----------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = Path("data")
DATA_ZIP = DATA_DIR / "smsspamcollection.zip"
DATA_TXT = DATA_DIR / "SMSSpamCollection"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def download_and_extract_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if DATA_TXT.exists():
        return
    st.info("Baixando dataset SMS Spam Collection...")
    urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    st.success("Dataset baixado e extraído.")


def load_dataset():
    if not DATA_TXT.exists():
        download_and_extract_dataset()
    df = pd.read_csv(DATA_TXT, sep="\t", header=None, names=["label", "text"])
    return df


def preprocess_text(s: pd.Series, remove_stopwords=False):
    # mínimo: lowercase and remove special characters
    import re
    s = s.str.lower()
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # stopwords optional (simple english stopwords)
    if remove_stopwords:
        from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
        s = s.apply(lambda t: " ".join([w for w in t.split() if w not in ENGLISH_STOP_WORDS]))
    return s


def train_vectorizers_and_models(X_train_text, y_train, use_tfidf=True, max_features=5000):
    if use_tfidf:
        vec = TfidfVectorizer(max_features=max_features)
    else:
        vec = CountVectorizer(max_features=max_features)
    X_train = vec.fit_transform(X_train_text)
    # Use Logistic Regression (linear) as default
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    return vec, clf


def evaluate_model(vec, clf, X_text, y_true):
    X = vec.transform(X_text)
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, y_pred

# ----------------------- Streamlit App -----------------------
st.set_page_config(page_title="Classificação Binária com LIME e SHAP", layout="wide")
st.title("Classificação Binária de Textos com LIME e SHAP")


# Sidebar "pages" for 5 integrantes
page = st.sidebar.selectbox("Selecionar página", (
    "Visão Geral (síntese)",
    "1 - Pipeline textual e comparação com tabular",
    "2 - Implementação dos modelos e métricas",
    "3 - Explicações com LIME",
    "4 - Explicações com SHAP",
    "5 - Síntese e Análise Crítica do Projeto"
))

with st.sidebar:
  
    st.subheader("Alunos:")
    st.markdown(""" \
    - Emanoel Sousa
    - José Macedo
    - Leonardo Vitorio
    - Marcos Araújo
    - Wadson Tardelle
    """
    )

# Load data once
with st.spinner("Carregando dataset..."):
    df = load_dataset()

# Encode labels: ham=0, spam=1
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])


if "results_compare" not in st.session_state:
    st.session_state.results_compare = []


# Show quick dataset preview
if page == "Visão Geral (síntese)":
    st.header("Resumo do projeto e estrutura do pipeline")
    st.markdown("""
    **Objetivo:** Construir um pipeline completo de classificação binária de textos (SMS) e explicar as previsões usando LIME e SHAP.

    **Etapas do pipeline:**
    1. Carregamento e pré-processamento de textos (lowercase, remover caracteres especiais, stopwords opcional).
    2. Representações: Bag-of-Words (CountVectorizer) e TF-IDF (TfidfVectorizer).
    3. Modelos lineares: Regressão Logística (ou SVM Linear).
    4. Avaliação: Accuracy, Precision, Recall, F1-score. Matriz de confusão e relatório.
    5. Interpretabilidade: LIME (local) e SHAP (global + local), análises comparativas.
    """)
    st.subheader("Preview do dataset")
    st.dataframe(df.sample(10, random_state=42))
    st.write("Distribuição das classes:")
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=df, ax=ax)
    ax.set_title('Count per class')
    st.pyplot(fig)
    st.markdown("\n---\nDeseja rodar experimentos? Vá para a página 'Integrante 2' para treinar modelos, 'Integrante 3' para LIME e 'Integrante 4' para SHAP.")

# ----------------------- Integrante 1 -----------------------
if page == "1 - Pipeline textual e comparação com tabular":
    st.header("1 — Pipeline textual e comparação com dados tabulares")
    
    st.markdown("**Exemplo prático:** mostraremos transformação mínima do texto e contagens de palavras (BOW) e TF-IDF)")

    remove_stop = st.checkbox("Remover stopwords (pré-processamento)", value=False)
    s = preprocess_text(df['text'], remove_stopwords=remove_stop)
    st.subheader("Exemplo de pré-processamento")
    st.table(pd.DataFrame({'orig': df['text'].head(6), 'preprocessed': s.head(6)}))

    st.subheader("Representações — Top tokens")
    max_feats = st.slider("max_features (vocab size)", 500, 10000, 2000)
    bow = CountVectorizer(max_features=max_feats)
    tfidf = TfidfVectorizer(max_features=max_feats)
    bow.fit(s)
    tfidf.fit(s)
    bow_top = pd.Series(bow.vocabulary_).sort_values()[:20]
    st.write("Vocabulário (amostra) — Bag-of-Words")
    st.write(list(list(bow.vocabulary_.keys())[:30]))
    st.write("TF-IDF (amostra)")
    st.write(list(list(tfidf.vocabulary_.keys())[:30]))

    st.markdown("\n**Comparação conceitual com dados tabulares:**\n- Textos exigem vetorização (BOW/TF-IDF) para virar features numéricas.\n    - Dados tabulares com colunas já numéricas/categóricas exigem imputação/normalização e codificação.\n    - Interpretação: em textos, features são tokens (palavras/ngrams) enquanto em tabular cada coluna já é uma feature legível.\n    ")

# ----------------------- Integrante 2 -----------------------


if page == "2 - Implementação dos modelos e métricas":
    st.header("2 — Implementação dos modelos (BOW/TF-IDF) e métricas")

    st.markdown("Escolha representação e treine modelos lineares. O treino pode demorar alguns segundos.")
    rep = st.radio("Representação", ("TF-IDF", "BOW"))
    max_feats = st.number_input("max_features", min_value=500, max_value=20000, value=5000, step=500)
    test_size = st.slider("test_size (fração)", 0.1, 0.5, 0.2)
    random_state = 42

    # preprocess
    s = preprocess_text(df['text'], remove_stopwords=False)
    X_train, X_test, y_train, y_test = train_test_split(s, df['label_enc'], test_size=test_size, stratify=df['label_enc'], random_state=random_state)

    use_tfidf = (rep == "TF-IDF")
    if st.button("Treinar modelo linear (LogisticRegression) e avaliar"):
        with st.spinner("Treinando..."):
            vec, clf = train_vectorizers_and_models(X_train, y_train, use_tfidf=use_tfidf, max_features=max_feats)
            joblib.dump(vec, MODEL_DIR / f"vectorizer_{rep}.pkl")
            joblib.dump(clf, MODEL_DIR / f"clf_logreg_{rep}.pkl")
        st.success("Treinamento concluído e modelos salvos em ./models/")

        metrics, y_pred = evaluate_model(vec, clf, X_test, y_test)
        st.subheader("Métricas de avaliação")
        st.json(metrics)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)
    
        metrics, y_pred = evaluate_model(vec, clf, X_test, y_test)

        result = {
            "Representação": rep,
            "Acurácia": metrics["accuracy"],
            "Precisão": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1"],
            "Falsos Positivos (FP)": int(fp),
            "Falsos Negativos (FN)": int(fn),
            "max_features": max_feats,
            "test_size": test_size
        }

        # Evita duplicação (BOW vs TF-IDF)
        st.session_state.results_compare = [
            r for r in st.session_state.results_compare
            if r["Representação"] != rep
        ]
        st.session_state.results_compare.append(result)

    if len(st.session_state.results_compare) >= 2:
        st.subheader("📊 Comparação entre BOW e TF-IDF")

        df_compare = pd.DataFrame(st.session_state.results_compare)
        df_compare = df_compare.set_index("Representação")

        st.dataframe(
            df_compare.style.format("{:.4f}")
                            .highlight_max(axis=0, color="green"),
            use_container_width=True
        )



# ----------------------- Integrante 3 -----------------------
if page == "3 - Explicações com LIME":
    st.header("3 — Explicações com LIME (local)")
    st.markdown("A seguir vamos carregar um modelo treinado (TF-IDF por padrão) e mostrar explicações locais com LIME para 3 exemplos: 2 corretos e 1 incorreto.")

    # try to load trained model
    rep_choice = st.selectbox("Carregar modelo treinado (representação)", ("TF-IDF", "BOW"))
    vec_path = MODEL_DIR / f"vectorizer_{rep_choice}.pkl"
    clf_path = MODEL_DIR / f"clf_logreg_{rep_choice}.pkl"

    if not vec_path.exists() or not clf_path.exists():
        st.warning("Modelos não encontrados em ./models/. Treine um modelo na página 'Integrante 2' antes de usar LIME.")
    else:
        vec = joblib.load(vec_path)
        clf = joblib.load(clf_path)

        # wrap classifier for LIME (expects raw text -> predict_proba)
        class_names = list(le.classes_)
        explainer = LimeTextExplainer(class_names=class_names)

        s = preprocess_text(df['text'], remove_stopwords=False)
        X_train, X_test, y_train, y_test = train_test_split(s, df['label_enc'], test_size=0.2, stratify=df['label_enc'], random_state=42)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # predict and find examples: two correct and one incorrect
        X_test_trans = vec.transform(X_test)
        y_pred = clf.predict(X_test_trans)
        correct_idx = np.where(y_pred == y_test)[0]
        wrong_idx = np.where(y_pred != y_test)[0]

        chosen = []
        if len(correct_idx) >= 2:
            chosen.extend(list(correct_idx[:2]))
        if len(wrong_idx) >= 1:
            chosen.append(int(wrong_idx[0]))

        st.write(f"Exemplos escolhidos (índices no conjunto de teste): {chosen}")

        for idx in chosen:
            st.markdown(f"---\n### Exemplo índice {idx}")
            text = X_test.iloc[idx]
            true = y_test.iloc[idx]
            pred = y_pred[idx]
            st.write(f"Texto: {text}")
            st.write(f"Rótulo verdadeiro: {le.inverse_transform([true])[0]} | Predito: {le.inverse_transform([pred])[0]}")

            # LIME explanation
            predict_proba = lambda texts: clf.predict_proba(vec.transform(texts))
            exp = explainer.explain_instance(text, predict_proba, num_features=10)
            st.subheader("Explicação LIME (palavras e pesos)")
            # Show as table
            exp_map = exp.as_list()
            df_exp = pd.DataFrame(exp_map, columns=["feature", "weight"]) 
            st.table(df_exp)

            # plot and show
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)

        # Aggregate LIME importances across a sample of instances for top 20
        st.subheader("Top 20 palavras (agregação LIME sobre 200 amostras)")
        sample_idx = np.random.choice(range(len(X_test)), size=min(200, len(X_test)), replace=False)
        agg = {}
        for i in sample_idx:
            text = X_test.iloc[i]
            num_feats = min(50, len(text.split()))
            tokens = text.split()
            doc_size = len(tokens)

            if doc_size < 2:
                st.warning("Mensagem muito curta para gerar explicação com LIME.")
                continue

            num_feats = min(20, doc_size)
            from lime.lime_text import LimeTextExplainer
            explainer_lime = LimeTextExplainer(class_names=['ham', 'spam'])
            exp = explainer_lime.explain_instance(
                text,
                predict_proba,
                num_features=num_feats,
                num_samples=1000
            )
            for feat, w in exp.as_list():
                # feat might be like 'word'=1; ensure consistent token
                token = feat
                agg[token] = agg.get(token, 0) + abs(w)
        agg_series = pd.Series(agg).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8,6))
        agg_series[::-1].plot.barh(ax=ax)
        ax.set_title('Top 20 palavras — LIME (soma dos pesos absolutos)')
        st.pyplot(fig)

# ----------------------- Integrante 4 -----------------------
if page == "4 - Explicações com SHAP":
    st.header("4 — Explicações com SHAP (global e local)")
    st.markdown("Carregue o mesmo modelo treinado e vamos gerar gráficos globais e locais com SHAP.\n**Atenção:** SHAP pode usar bastante memória para matrizes grandes. Recomendado usar modelo TF-IDF com max_features pequeno (ex: 5000).")

    rep_choice = st.selectbox("Carregar modelo treinado (representação)", ("TF-IDF", "BOW"), key="shap_rep")
    vec_path = MODEL_DIR / f"vectorizer_{rep_choice}.pkl"
    clf_path = MODEL_DIR / f"clf_logreg_{rep_choice}.pkl"

    if not vec_path.exists() or not clf_path.exists():
        st.warning("Modelos não encontrados em ./models/. Treine um modelo na página 'Integrante 2' antes de usar SHAP.")
    else:
        vec = joblib.load(vec_path)
        clf = joblib.load(clf_path)

        s = preprocess_text(df['text'], remove_stopwords=False)
        X_train, X_test, y_train, y_test = train_test_split(s, df['label_enc'], test_size=0.2, stratify=df['label_enc'], random_state=42)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Transform to numeric
        X_train_mat = vec.transform(X_train)
        X_test_mat = vec.transform(X_test)

        st.subheader("SHAP — Global summary plot (top features)")
        # For linear models, use LinearExplainer for efficiency
        with st.spinner("Calculando valores SHAP (pode demorar)..."):
            try:
                explainer = shap.LinearExplainer(clf, X_train_mat, feature_perturbation="interventional")
            except Exception as e:
                # fallback
                explainer = shap.Explainer(clf, X_train_mat)
            shap_values = explainer.shap_values(X_test_mat)

        # shap_values shape: (n_classes, n_samples, n_features) for multiclass or (n_samples, n_features)
        # For binary logistic regression sklearn returns shape (n_samples, n_features)
        # We'll compute mean absolute shap per feature
        if isinstance(shap_values, list):
            # multiclass -> take class 1
            sv = shap_values[1]
        else:
            sv = shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        feature_names = np.array(vec.get_feature_names_out())
        top20_idx = np.argsort(mean_abs)[-20:][::-1]
        top20 = pd.Series(mean_abs[top20_idx], index=feature_names[top20_idx])

        fig, ax = plt.subplots(figsize=(8,6))
        top20[::-1].plot.barh(ax=ax)
        ax.set_title('Top 20 palavras — SHAP (mean |SHAP value|)')
        st.pyplot(fig)

        st.markdown("\n**SHAP summary plot (pontos)**")
        # show summary plot
        try:
            fig2 = plt.figure()
            shap.summary_plot(sv, features=X_test_mat, feature_names=feature_names, show=False, max_display=20)
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Não foi possível gerar o summary_plot interativo: {e}")

        st.subheader("SHAP — Explicação local (waterfall) para um exemplo)")
        idx = st.number_input("Índice do exemplo no conjunto de teste (0..)", min_value=0, max_value=max(0, len(X_test)-1), value=0)
        # Compute and plot waterfall
        try:
            shap_val_example = sv[idx]
            base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else explainer.expected_value[1]
            fig3 = plt.figure(figsize=(6,4))
            shap.plots.waterfall(shap.Explanation(values=shap_val_example, base_values=base_value, data=X_test_mat[idx].toarray().ravel(), feature_names=feature_names), show=False)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Erro ao gerar waterfall: {e}")

        # Compare top 20 from LIME and SHAP if LIME agg file exists in memory (we'll compute LIME agg on the fly but warn about time)
        if st.checkbox("Gerar agregação LIME (para comparação com SHAP) — pode demorar"):
            explainer_lime = LimeTextExplainer(class_names=list(le.classes_))
            sample_idx = np.random.choice(range(len(X_test)), size=min(200, len(X_test)), replace=False)
            agg = {}
            predict_proba = lambda texts: clf.predict_proba(vec.transform(texts))
            with st.spinner("Computando LIME em 200 amostras..."):
                for i in sample_idx:
                    text = X_test.iloc[i]
                    num_feats = min(10, len(text.split()))
                    exp = explainer_lime.explain_instance(text, predict_proba, num_features=num_feats)
                    for feat, w in exp.as_list():
                        agg[feat] = agg.get(feat, 0) + abs(w)
            agg_series = pd.Series(agg).sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8,6))
            agg_series[::-1].plot.barh(ax=ax)
            ax.set_title('Top 20 palavras — LIME (soma dos pesos absolutos)')
            st.pyplot(fig)

# ----------------------- Integrante 5 -----------------------
if page == "5 - Síntese e Análise Crítica do Projeto":
    st.header("5 — Síntese e Análise Crítica do Projeto")

    st.markdown("""
    ## 🎯 Objetivo Geral
    Este projeto integrou pré-processamento textual, modelos supervisionados lineares
    e técnicas de interpretabilidade (LIME e SHAP) aplicados ao dataset *SMS Spam Collection*.

    ## 🧠 Principais Contribuições
    - Construção de um **pipeline completo** de classificação binária.
    - Comparação entre duas formas de vetorização:
      - **Bag-of-Words**
      - **TF-IDF**
    - Avaliação de desempenho usando métricas fundamentais:
      - Accuracy
      - Precision
      - Recall
      - F1-score
    - Interpretação das previsões com:
      - **LIME**: interpretabilidade local
      - **SHAP**: interpretabilidade global e local

    ## 📈 Análises Críticas
    ### 1. Sobre o dataset
    - Contém textos curtos e altamente redundantes.
    - Variação limitada de vocabulário reduz impacto de modelos complexos.
    - Classe "spam" é minoritária, exigindo cuidado com métricas.

    ### 2. Sobre os modelos
    - Modelos lineares (LogisticRegression, LinearSVC) são adequados.
    - TF-IDF apresentou geralmente desempenho superior ao BOW.
    - Modelos mais complexos (transformers) não seriam necessários para este trabalho.

    ### 3. Sobre LIME
    - Explicações intuitivas, úteis para textos curtos.
    - Limitações:
      - Dependência da perturbação aleatória.
      - Instabilidade quando o texto tem poucas palavras.
      - Tempo de execução maior em análises globais.

    ### 4. Sobre SHAP
    - Oferece visão global mais estável.
    - Requer mais memória e apresenta maior custo computacional.
    - Excelente para discutir transparência e ética em IA.

    ### 5. Comparação LIME vs SHAP
    | Critério | LIME | SHAP |
    |---------|------|-------|
    | Foco | Local | Global + Local |
    | Estabilidade | Média | Alta |
    | Interpretação | Muito intuitiva | Pode ser mais complexa |
    | Custo computacional | Médio | Alto |
    | Melhor uso | Explicar previsão individual | Analisar importância geral |

    ## 📝 Conclusão Geral
    O projeto demonstrou que:
    - Modelos lineares continuam extremamente eficientes para problemas de spam.
    - Escolhas de vetorização influenciam mais do que o tipo de modelo.
    - Ferramentas de interpretabilidade são essenciais para justificar decisões de IA.
    - A união entre desempenho e interpretabilidade deve sempre ser incentivada
      em ambientes educacionais e profissionais.

    ## 📌 Sugestões de Trabalhos Futuros
    - Adicionar embeddings como Word2Vec, GloVe ou FastText.
    - Testar Naive Bayes otimizado para texto.
    - Comparar com modelos baseados em Transformers.
    - Criar gráficos interativos usando Plotly ou Streamlit-AgGrid.
    - Aplicar o pipeline a outros tipos de texto, como fake news ou sentiment analysis.
    """)

    st.success("Página 5 carregada com sucesso!")



