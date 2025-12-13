# 📊 Classificação Binária de SMS com LIME e SHAP

Aplicação completa em **Streamlit** para estudo de *Machine Learning
explicável* utilizando o dataset **SMS Spam Collection**.

------------------------------------------------------------------------

## 🧩 Visão Geral do Projeto

Este projeto implementa todo o pipeline de classificação binária de
textos:

-   📥 **Carregamento do dataset**
-   🧹 **Pré-processamento**
-   🔤 **Vetorização** (Bag-of-Words e TF-IDF)
-   🤖 **Treinamento de modelos**
-   📈 **Avaliação (accuracy, precision, recall, f1-score)**
-   🔍 **Explicabilidade com LIME**
-   🔵 **Explicabilidade global e local com SHAP**
-   🖥️ **Interface amigável construída em Streamlit**

------------------------------------------------------------------------

## 📘 Estrutura do Projeto

    📂 trabalho_final_streamlit/
    │── app.py
    │── requirements.txt
    │── README.md
    │── /models
    │── /data
    │── /pages   (5 páginas do projeto)

------------------------------------------------------------------------

## 🚀 Executando o Projeto

### 1️⃣ Clonar o repositório

``` bash
git clone https://github.com/marcosaraujo2020/projeto_final_topicos_especial_programacao_2025.2.git
cd seu-repo
```

### 2️⃣ Criar ambiente virtual

``` bash
python3 -m venv .venv
source .venv/bin/activate  # Linux
```

### 3️⃣ Instalar dependências

``` bash
pip install -r requirements.txt
```

### 4️⃣ Executar a aplicação

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📦 Dependências Principais

-   streamlit\
-   scikit-learn\
-   pandas\
-   numpy\
-   lime\
-   shap\
-   joblib

------------------------------------------------------------------------

## 🖼️ Demonstração das Funcionalidades

### 📌 Página 1 -- Introdução ao Dataset

Mostra amostra dos dados e distribuição entre *spam* e *ham*.

### 📌 Página 2 -- Vetorização

Demonstra TF‑IDF e Bag‑of‑Words com gráficos.

### 📌 Página 3 -- Treinamento e Avaliação

Modelos, métricas e matriz de confusão.

### 📌 Página 4 -- Explicações LIME & SHAP

Explicações locais e globais, comparação entre top‑features.

### 📌 Página 5 -- Síntese Crítica

Discussão final do grupo + reflexão técnica.

------------------------------------------------------------------------

## 🧠 Insights Importantes

-   TF‑IDF melhora consideravelmente o desempenho\
-   LIME destaca palavras específicas da mensagem\
-   SHAP fornece impacto global das features\
-   Ambos convergem para tokens como **free**, **call**, **txt**,
    **now**

------------------------------------------------------------------------

## 🔧 Personalização

Sinta‑se livre para:

-   Trocar modelo (SVM, RandomForest, Naive Bayes)
-   Incluir *n‑grams*
-   Implementar balanceamento de classes
-   Criar novas visualizações no Streamlit

------------------------------------------------------------------------

## 👨‍🏫 Projeto Acadêmico

Desenvolvido como trabalho final da disciplina\
**Tópicos Especiais em Programação**\
com foco em **Machine Learning explicável**.

------------------------------------------------------------------------

## 📄 Licença

MIT -- sinta-se livre para usar e expandir!

------------------------------------------------------------------------
