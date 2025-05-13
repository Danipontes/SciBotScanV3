import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import math
import textstat
import emoji
import numpy as np

from collections import Counter
from difflib import SequenceMatcher
from unidecode import unidecode
from datetime import datetime
from io import BytesIO
from textblob import TextBlob
from matplotlib.patches import Wedge, Circle

# ----------------------------------
# Carregar artefatos gerados na tese
# ----------------------------------
xgb_model = joblib.load("modelo_xgboost.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# --------------------
# Engenharia de features
# --------------------
def count_digits(text): return sum(c.isdigit() for c in str(text))
def count_letters(text): return sum(c.isalpha() for c in str(text))
def count_special_chars(text): return sum(not c.isalnum() for c in str(text))
def entropy(text):
    if not text: return 0
    prob = [n_x / len(text) for x, n_x in Counter(text).items()]
    return -sum(p * math.log2(p) for p in prob)
def has_repeated_chars(text): return int(any(text[i] == text[i+1] for i in range(len(text)-1)))
def is_abbreviation(full_name, username):
    initials = ''.join([word[0] for word in full_name.split() if word])
    return int(initials.lower() in username.lower())
def jaccard_similarity(a, b):
    a_set = set(a.lower())
    b_set = set(b.lower())
    union = a_set.union(b_set)
    intersection = a_set.intersection(b_set)
    return len(intersection) / len(union) if union else 0
def levenshtein_ratio(s, t): return SequenceMatcher(None, s, t).ratio()
def count_emojis(text): return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)

def gerar_features(df_contas, df_posts):
    df = df_posts.copy()
    df["QTDSEGUIDORES"] = pd.to_numeric(df["QTDSEGUIDORES"], errors='coerce')
    df["CONTEUDOPOST"] = df["CONTEUDOPOST"].astype(str)
    df["DATATEMPO"] = pd.to_datetime(df["DATATEMPO"])

    # Métricas por post
    df["num_caracteres"] = df["CONTEUDOPOST"].str.len()
    df["num_palavras"] = df["CONTEUDOPOST"].str.split().str.len()
    df["palavras_unicas"] = df["CONTEUDOPOST"].apply(lambda x: len(set(str(x).split())))
    df["hora"] = df["DATATEMPO"].dt.hour
    df["dia_da_semana"] = df["DATATEMPO"].dt.weekday
    df["tempo_entre_posts"] = df.groupby("ACCOUNT")["DATATEMPO"].diff().dt.total_seconds() / 60
    df["hashtags"] = df["CONTEUDOPOST"].str.count("#")
    df["links"] = df["CONTEUDOPOST"].str.count("http")
    df["mentions"] = df["CONTEUDOPOST"].str.count("@")
    df["exclamacoes"] = df["CONTEUDOPOST"].str.count("!")
    df["interrogacoes"] = df["CONTEUDOPOST"].str.count(r"\?")
    df["maiusculas"] = df["CONTEUDOPOST"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    df["pontuacao_excessiva"] = df["CONTEUDOPOST"].str.count(r"\.\.\.|!!!")
    df["num_emojis"] = df["CONTEUDOPOST"].apply(count_emojis)
    df["sentimento"] = df["CONTEUDOPOST"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["post_repetido"] = df.duplicated(subset=["CONTEUDOPOST"], keep=False)

    termos_cientificos = ["pesquisa", "artigo", "científico", "universidade", "research", "article", "paper", "scientific", "university", "published", "preprint"]
    df["palavras_cientificas"] = df["CONTEUDOPOST"].apply(lambda x: sum(1 for palavra in termos_cientificos if palavra in str(x).lower()))
    df["legibilidade"] = df["CONTEUDOPOST"].apply(lambda x: textstat.flesch_reading_ease(str(x)))

    # Agregações por conta
    df_agg = df.groupby("ACCOUNT").agg(media_seguidores=("QTDSEGUIDORES", "mean"),
                                       tamanho_medio_post=("CONTEUDOPOST", lambda x: x.str.len().mean())).reset_index()

    df_text_features = df.groupby("ACCOUNT").agg(
        diversidade_vocabulario=("palavras_unicas", lambda x: x.mean() / x.max()),
        media_hashtags=("hashtags", "mean"),
        media_links=("links", "mean"),
        media_mentions=("mentions", "mean"),
        media_exclamacoes=("exclamacoes", "mean"),
        media_interrogacoes=("interrogacoes", "mean"),
        taxa_maiusculas=("maiusculas", "mean"),
        media_sentimento=("sentimento", "mean"),
        taxa_posts_repetidos=("post_repetido", "mean"),
        media_palavras_cientificas=("palavras_cientificas", "mean"),
        media_legibilidade=("legibilidade", "mean")
    ).reset_index()

    df_temporais = df.groupby("ACCOUNT").agg(
        media_hora=("hora", "mean"),
        desvio_hora=("hora", "std"),
        media_dia_semana=("dia_da_semana", "mean"),
        media_tempo_entre_posts=("tempo_entre_posts", "mean"),
        desvio_tempo_entre_posts=("tempo_entre_posts", "std"),
        posts_dia_util=("dia_da_semana", lambda x: (x < 5).sum()),
        posts_fim_de_semana=("dia_da_semana", lambda x: (x >= 5).sum())
    ).reset_index()

    df_mais_features = df.groupby("ACCOUNT").agg(
        media_caracteres=("num_caracteres", "mean"),
        media_palavras=("num_palavras", "mean"),
        media_pontuacao_excessiva=("pontuacao_excessiva", "mean"),
        media_emojis=("num_emojis", "mean")
    ).reset_index()

    # Nome e username
    df_nome_features = df.drop_duplicates(subset="ACCOUNT").copy()
    df_nome_features["tamanho_name"] = df_nome_features["NOME"].astype(str).apply(len)
    df_nome_features["quantidade_palavras_name"] = df_nome_features["NOME"].astype(str).apply(lambda x: len(x.split()))
    df_nome_features["maioria_maiuscula_name"] = df_nome_features["NOME"].astype(str).apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0)
    df_nome_features["contém_números_name"] = df_nome_features["NOME"].astype(str).apply(lambda x: int(any(c.isdigit() for c in x)))
    df_nome_features["entropia_name"] = df_nome_features["NOME"].astype(str).apply(entropy)
    df_nome_features["contém_emoji_name"] = df_nome_features["NOME"].astype(str).apply(lambda x: int(any(char in emoji.EMOJI_DATA for char in x)))
    df_nome_features["tamanho_username"] = df_nome_features["ACCOUNT"].astype(str).apply(len)
    df_nome_features["digitos_username"] = df_nome_features["ACCOUNT"].astype(str).apply(count_digits)
    df_nome_features["letras_username"] = df_nome_features["ACCOUNT"].astype(str).apply(count_letters)
    df_nome_features["especial_username"] = df_nome_features["ACCOUNT"].astype(str).apply(count_special_chars)
    df_nome_features["proporcao_digitos_username"] = df_nome_features["digitos_username"] / df_nome_features["tamanho_username"]
    df_nome_features["tem_numeros_fim_username"] = df_nome_features["ACCOUNT"].astype(str).apply(lambda x: int(bool(re.search(r'\\d+$', x))))
    df_nome_features["contém_bot_username"] = df_nome_features["ACCOUNT"].astype(str).apply(lambda x: int("bot" in x.lower()))
    df_nome_features["entropia_username"] = df_nome_features["ACCOUNT"].astype(str).apply(entropy)
    df_nome_features["tem_repeticoes_username"] = df_nome_features["ACCOUNT"].astype(str).apply(has_repeated_chars)
    df_nome_features["similaridade_jaccard"] = df_nome_features.apply(lambda x: jaccard_similarity(str(x["NOME"]), str(x["ACCOUNT"])), axis=1)
    df_nome_features["similaridade_levenshtein"] = df_nome_features.apply(lambda x: levenshtein_ratio(str(x["NOME"]), str(x["ACCOUNT"])), axis=1)
    df_nome_features["username_é_abreviação"] = df_nome_features.apply(lambda x: is_abbreviation(str(x["NOME"]), str(x["ACCOUNT"])), axis=1)

    # Merge com dados das contas
    df_merged = df_nome_features.merge(df_contas, on="ACCOUNT", how="left")
    df_merged = df_merged.merge(df_agg, on="ACCOUNT", how="left")
    df_merged = df_merged.merge(df_text_features, on="ACCOUNT", how="left")
    df_merged = df_merged.merge(df_temporais, on="ACCOUNT", how="left")
    df_merged = df_merged.merge(df_mais_features, on="ACCOUNT", how="left")

    return df_merged.drop(columns=["ACCOUNT", "NOME"], errors="ignore")

def gauge_chart(porcentagem):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    # Arco colorido
    for i in range(100):
        ang1 = 180 * i / 100
        ang2 = 180 * (i + 1) / 100
        color = plt.cm.RdYlGn(i / 100)
        wedge = Wedge(center=(0, 0), r=1, theta1=ang1, theta2=ang2, width=0.3, color=color)
        ax.add_patch(wedge)

    # Ponteiro
    ang_pointer = np.radians(180 * porcentagem / 100)
    x = 0.85 * np.cos(np.pi - ang_pointer)
    y = 0.85 * np.sin(np.pi - ang_pointer)
    ax.plot([0, x], [0, y], color='black', linewidth=3)

    # Círculo central
    circle = Circle((0, 0), 0.05, color='black')
    ax.add_patch(circle)

    # Texto da porcentagem
    ax.text(0, -0.1, f"{porcentagem:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold')

    return fig

# ----------------------
# Classificador XGBOOST
# ----------------------
def preprocessar_e_classificar(df_features):
    X_imp = imputer.transform(df_features)
    X_scaled = scaler.transform(X_imp)
    prob = xgb_model.predict_proba(X_scaled)[0][1]
    pred = "BOT" if prob >= 0.82 else "HUMANO"
    return pred, prob, X_scaled

# --------------------
# Interface Streamlit
# --------------------
st.set_page_config(page_title="SciBotScan", layout="centered")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("SciBotScan_logo.png", width=300)
st.markdown("<h1 style='text-align: center; color: #004080;'>Welcome to SciBotScan</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>A bot detection tool for academic article postings</h4>", unsafe_allow_html=True)
st.markdown("---")

# --------------------
# Início do formulário estilizado
# --------------------
with st.container():
    st.markdown("""
        <div class="input-section">
            <h3 style='text-align:center; color: #004080; margin-bottom: 0;'>🔍 Classifique uma nova conta 🤖 vs 👤</h3>
            </div>
    """, unsafe_allow_html=True)
    st.subheader("Insira os dados da conta e dos posts")
    
    # Organizar em duas colunas
    col1, col2, col3 = st.columns(3)
    with col1:
        nome = st.text_input("Nome do perfil")
        qtd_posts = st.number_input("Quantidade total de posts", min_value=1)
        qtd_grandearea = st.number_input("Quantidade de grandes áreas", min_value=1)
        
    with col2:
        account = st.text_input("Account")
        qtd_doi = st.number_input("Quantidade de DOIs compartilhados", min_value=1)
        qtd_subarea = st.number_input("Quantidade de subáreas", min_value=1)

    with col3:
        qtd_seguidores = st.number_input("Quantidade de Seguidores", min_value=0)
        
    # Organizar em duas colunas
    c1, c2 = st.columns(2)
    with c1:
    # Dados dos posts (df_posts)
        horarios = st.text_area("Horários das postagens (ex: 15:45), uma por linha")
    with c2:
        datas = st.text_area("Datas das postagens (ex: 2024-05-08), uma por linha")

    postagens = st.text_area(f"Cole a(s) {qtd_posts} postagem(ns), separada(s) por quebra de linha")


    if st.button("Get Started 🚀"):
        lista_posts = [p.strip() for p in postagens.split('\n') if p.strip()]
        lista_datas = [d.strip() for d in datas.split('\n') if d.strip()]
        lista_horas = [h.strip() for h in horarios.split('\n') if h.strip()]

        if len(lista_posts) == len(lista_datas) == len(lista_horas) and len(lista_posts) == qtd_posts: # tem que ser igual a quantidade de posts

            # 1. Criar df_contas
            df_contas = pd.DataFrame([{
                "ACCOUNT": account,
                "QTDPOSTS": qtd_posts,
                "QTDSUBAREA": qtd_subarea,
                "QTDGRANDEAREA": qtd_grandearea,
                "QTDDOIS": qtd_doi
            }])

            # 2. Criar df_posts
            df_posts = pd.DataFrame({
                "NOME": [nome] * len(lista_posts),
                "ACCOUNT": [account] * len(lista_posts),
                "QTDSEGUIDORES": [qtd_seguidores] * len(lista_posts),
                "CONTEUDOPOST": lista_posts,
                "DATATEMPO": [f"{d} {h}" for d, h in zip(lista_datas, lista_horas)]
            })
            try:
                df_features = gerar_features(df_contas, df_posts)
                top_features = joblib.load("top_46_features.pkl")
                df_features = df_features.reindex(columns=top_features, fill_value=0)
                classe, probabilidade, X_scaled = preprocessar_e_classificar(df_features)
                percentual = probabilidade * 100
                st.success(f"Classe prevista: {classe} \nProbabilidade: {percentual:.2f}")
                
                # Interface Streamlit               
                st.subheader("Visualização da probabilidade - Gráfico Meia-Lua")
                fig = gauge_chart(percentual)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Erro ao processar os dados: {e}")
        else:
            st.error("As listas de postagens, datas e horários devem ter o mesmo tamanho e conter o numero de registros igual a quantidade de posts.")

# --------------------
# Cabeçalho institucional (estilo acadêmico visualmente elegante)
# --------------------
with st.container():
    st.markdown("""
    <style>
    .title-style {
        font-size: 2.6rem;
        font-weight: 800;
        color: #002244;
        margin: 2rem 0 1rem 0;
        font-family: 'Arial', sans-serif;
    }
    .intro-style {
        font-size: 1.05rem;
        color: #FFFFFF;
        line-height: 1.8;
        text-align: justify;
        max-width: 1000px;
        margin: auto;
        font-family: 'Calibri', sans-serif;
    }
    .intro-style ul {
        padding-left: 1.2rem;
    }
    .intro-style li {
        margin-bottom: 0.5rem;
    }
    .highlight {
        font-weight: bold;
        color: #005588;
    }
    .input-section {
        background-color: #f9f9f9;
        padding: 2rem;
        margin-top: 3rem;
        border-radius: 8px;
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
        font-family: 'Arial', sans-serif;
    }
    .input-section {
        background-color: #f9f9f9;
        margin-top: 2rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        margin: 10px;
        max-width: 1000px;
        text-align: center;
    }
    .input-section h3 {
        color: #003366;
        margin-bottom: 0;
        margin-top: 0
    }
    </style>
                
    <div class="intro-style">
        <strong style="font-size: 1.5em;">Sobre o SciBotScan: Desenvolvimento e Desempenho do Modelo:</strong><br><br>
        O SciBotScan é um modelo de inteligência artificial desenvolvido com base em um rigoroso processo de rotulagem e classificação de contas da plataforma X (antigo Twitter), com o objetivo de identificar contas humanas e bots que divulgam artigos científicos. A base de dados foi construída por meio da integração de algoritmos automáticos e fontes reconhecidas da literatura, complementada por uma verificação manual de mais de 13 mil contas. Ao final desse processo, foram identificadas 822 contas de bots e 12.945 contas humanas, com mais de 67 mil postagens analisadas. O dataset rotulado está disponível em: xxxxxx.<br><br>
        O modelo de classificação utiliza o algoritmo XGBoost e foi treinado com 46 features preditivas, considerando características de atividade, textualidade, comportamento temporal e estrutura dos nomes de usuário. Os principais indicadores de desempenho obtidos foram:
    </div>
                
    <div class="intro-style">
                
        📊
        AUC ROC: 0,9392                
        Kappa de Cohen: 0,5175
        Acurácia geral: 94,29%
        
    </div> 
                
    <div class="intro-style">           
        A análise de importância das variáveis (<strong>SHAP</strong>) revelou que os fatores mais relevantes para a predição incluem: número total de postagens, frequência entre postagens, uso de exclamações, sentimento textual e número de seguidores.<br><br>
        Todos os artefatos do modelo — classificador treinado, scaler, imputador, threshold e lista de features — foram salvos e podem ser utilizados diretamente nesta interface para classificação automática de novas contas. Acesse: xxxxxxxxxxxxxxxxxxxxx.
    </div>
                
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<center><small>Identify bots in your field of research and keep the scholarly community engaged.</small></center>", unsafe_allow_html=True)
