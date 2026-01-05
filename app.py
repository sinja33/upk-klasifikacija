# Streamlit App for Text Classification Project

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os
from pathlib import Path

import config
import utils
import broader_categories

# Configure page
st.set_page_config(
    page_title="Tekstovna Klasifikacija",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# LOAD DATA AND MODELS

@st.cache_resource
def load_all_data():
    """Load all datasets and models."""
    try:
        # Load original dataset
        df, category_names = utils.load_or_download_data()
        
        # Create broader categories version
        df_broader, broader_cats = broader_categories.create_broader_dataset(df, category_names)
        
        # Load Slovenian dataset
        slovenian_path = 'data/slovenian_news_placeholder.csv'
        if os.path.exists(slovenian_path):
            df_slovenian = pd.read_csv(slovenian_path, encoding='utf-8')
            slovenian_cats = sorted(df_slovenian['category'].unique())
        else:
            df_slovenian = None
            slovenian_cats = []
        
        # Load models
        models_dir = Path('data/models')
        models = {}
        
        if models_dir.exists():
            for model_file in models_dir.glob('*.pkl'):
                if model_file.stem != 'training_summary':
                    with open(model_file, 'rb') as f:
                        models[model_file.stem] = pickle.load(f)
        
        # Load training summary
        summary_path = models_dir / 'training_summary.pkl'
        if summary_path.exists():
            with open(summary_path, 'rb') as f:
                training_summary = pickle.load(f)
        else:
            training_summary = None
        
        # Load vectorizer comparison results
        vec_results = None
        if os.path.exists('data/vectorizer_results.pkl'):
            with open('data/vectorizer_results.pkl', 'rb') as f:
                vec_results = pickle.load(f)
        
        return {
            'df_20news': df,
            'categories_20news': category_names,
            'df_broader': df_broader,
            'categories_broader': broader_cats,
            'df_slovenian': df_slovenian,
            'categories_slovenian': slovenian_cats,
            'models': models,
            'training_summary': training_summary,
            'vectorizer_results': vec_results
        }
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None



# HELPER FUNCTIONS

def get_top_keywords_for_categories(vectorizer, model, category_names, n_words=5):
    """
    Pridobi top N besed za vsako kategorijo.
    
    Args:
        vectorizer: Fitted vectorizer
        model: Trained model (mora imeti coef_ ali feature_log_prob_)
        category_names: Lista imen kategorij
        n_words: Število besed za vsako kategorijo
        
    Returns:
        DataFrame z top besedami za vsako kategorijo
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()
    
    keywords_data = []
    
    if hasattr(model, 'coef_'):
        # Linear SVM ali Logistic Regression
        if len(category_names) == 2:
            # Binary classification
            coef = model.coef_[0]
            
            # Pozitivne (razred 1)
            top_pos_idx = np.argsort(coef)[-n_words:][::-1]
            keywords_data.append({
                'Kategorija': category_names[1],
                'Top Besede': ', '.join([f"{feature_names[i]}" for i in top_pos_idx]),
                'Koeficienti': ', '.join([f"{coef[i]:.3f}" for i in top_pos_idx])
            })
            
            # Negativne (razred 0)
            top_neg_idx = np.argsort(coef)[:n_words]
            keywords_data.append({
                'Kategorija': category_names[0],
                'Top Besede': ', '.join([f"{feature_names[i]}" for i in top_neg_idx]),
                'Koeficienti': ', '.join([f"{coef[i]:.3f}" for i in top_neg_idx])
            })
        else:
            # Multi-class
            for i, category in enumerate(category_names):
                coef = model.coef_[i]
                top_idx = np.argsort(coef)[-n_words:][::-1]
                
                keywords_data.append({
                    'Kategorija': category,
                    'Beseda 1': f"{feature_names[top_idx[0]]} ({coef[top_idx[0]]:.3f})",
                    'Beseda 2': f"{feature_names[top_idx[1]]} ({coef[top_idx[1]]:.3f})",
                    'Beseda 3': f"{feature_names[top_idx[2]]} ({coef[top_idx[2]]:.3f})",
                    'Beseda 4': f"{feature_names[top_idx[3]]} ({coef[top_idx[3]]:.3f})",
                    'Beseda 5': f"{feature_names[top_idx[4]]} ({coef[top_idx[4]]:.3f})"
                })
    
    elif hasattr(model, 'feature_log_prob_'):
        # Naive Bayes
        for i, category in enumerate(category_names):
            log_probs = model.feature_log_prob_[i]
            top_idx = np.argsort(log_probs)[-n_words:][::-1]
            
            keywords_data.append({
                'Kategorija': category,
                'Beseda 1': f"{feature_names[top_idx[0]]} ({log_probs[top_idx[0]]:.2f})",
                'Beseda 2': f"{feature_names[top_idx[1]]} ({log_probs[top_idx[1]]:.2f})",
                'Beseda 3': f"{feature_names[top_idx[2]]} ({log_probs[top_idx[2]]:.2f})",
                'Beseda 4': f"{feature_names[top_idx[3]]} ({log_probs[top_idx[3]]:.2f})",
                'Beseda 5': f"{feature_names[top_idx[4]]} ({log_probs[top_idx[4]]:.2f})"
            })
    
    if keywords_data:
        return pd.DataFrame(keywords_data)
    else:
        return None


# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================

def page_project_overview(data):
    """Stran 1: Pregled Projekta."""
    
    st.title("Tekstovna analiza in klasifikacija dokumentov po kategorijah")
    st.markdown("Celovita klasifikacija besedil z uporabo strojnega učenja")
    
    st.markdown("---")
    
    # Ključna statistika
    st.subheader("Ključna Statistika")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nabori Podatkov", "3")
        st.caption("20 News (Vsi), Širši, Slovenski")
    
    with col2:
        st.metric("Skupaj Dokumentov", f"{len(data['df_20news']):,}")
        st.caption("Vzorci za treniranje")
    
    with col3:
        st.metric("ML Modeli", "3")
        st.caption("Naive Bayes, LogReg, SVM")
    
    with col4:
        if data['training_summary']:
            best_acc = max([v['test_score'] for v in data['training_summary'].values()])
            st.metric("Najboljša Točnost", f"{best_acc:.1%}")
            st.caption("Linear SVM (Širši)")
    
    st.markdown("---")
    
    # Kaj je tekstovna klasifikacija
    st.subheader("Kaj je Tekstovna Klasifikacija?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Tekstovna klasifikacija** je nadzorovan postopek strojnega učenja, ki besedilnim dokumentom 
        dodeljuje vnaprej določene kategorije.
        
        **Aplikacije v praksi:**
        - Zaznavanje neželene pošte
        - Kategorizacija novic
        - Moderiranje vsebin
        
        **Ta projekt demonstrira:**
        - Več metod vektorizacije besedil
        - Celovito primerjavo modelov
        - Evalvacijo na treh podatkovnih naborih
        - Večjezično testiranje (angleščina in slovenščina)
        """)
    
    with col2:
        st.info("""
        **Proces:**
        1. Surovo besedilo
        2. Predprocesiranje
        3. Vektorizacija
        4. Treniranje modela
        5. Napoved kategorij
        """)
    
    st.markdown("---")
    
    # Cilji projekta
    st.subheader("Cilji Projekta")
    
    goals = [
        "Primerjava različnih metod vektorizacije besedil (TF-IDF, Count, Hashing)",
        "Treniranje in optimizacija več ML modelov z Grid Search",
        "Evalvacija uspešnosti na različnih podatkovnih naborih",
        "Izdelava interaktivnega vmesnika za raziskovanje modelov"
    ]
    
    for i, goal in enumerate(goals, 1):
        st.markdown(f"**{i}.** {goal}")
    
    st.markdown("---")
    
    # Ključni rezultati
    if data['training_summary']:
        st.subheader("Ključni Rezultati")
        
        # Najboljši model za vsak dataset
        datasets_info = {
            '20news_all': '20 Newsgroups (Vsi)',
            '20news_broader': '20 Newsgroups (Širši)',
            'slovenian': 'Slovenski Teksti'
        }
        
        for dataset_key, dataset_name in datasets_info.items():
            dataset_models = {k: v for k, v in data['training_summary'].items() if k.startswith(dataset_key)}
            
            if dataset_models:
                best_model = max(dataset_models.items(), key=lambda x: x[1]['test_score'])
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.markdown(f"**{dataset_name}**")
                
                with col2:
                    model_name = best_model[0].replace(dataset_key + '_', '').replace('_', ' ')
                    st.write(f"Najboljši: {model_name}")
                
                with col3:
                    st.write(f"**{best_model[1]['test_score']:.1%}** točnost")


# ==========================================
# PAGE 2: DATA & PREPROCESSING
# ==========================================

def page_data_preprocessing(data):
    """Stran 2: Podatki in Predprocesiranje."""
    
    st.title("Podatki in Predprocesiranje")
    
    # Zavihki za datasete
    tab1, tab2, tab3 = st.tabs(["20 Newsgroups (Vsi)", "20 Newsgroups (Širši)", "Slovenski Teksti"])
    
    # Zavihek 1: 20 Newsgroups Vsi
    with tab1:
        st.subheader("20 Newsgroups Dataset (Vseh 20 Kategorij)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Skupaj Dokumentov", f"{len(data['df_20news']):,}")
        with col2:
            st.metric("Kategorij", len(data['categories_20news']))
        with col3:
            avg_length = data['df_20news']['clean_text'].str.len().mean()
            st.metric("Povprečna Dolžina", f"{avg_length:.0f} znakov")
        
        st.markdown("---")
        
        # Prikaz kategorij
        st.markdown("**Kategorije**")
            
        for cat in data['categories_20news']:
            st.markdown(f"- {cat}")
        
        st.markdown("---")
        
        # Distribucija kategorij
        st.markdown("**Distribucija Kategorij**")
        
        cat_counts = data['df_20news']['category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(cat_counts.index, cat_counts.values, color='#2196F3')
        ax.set_xlabel('Število Dokumentov', fontweight='bold')
        ax.set_ylabel('Kategorija', fontweight='bold')
        ax.set_title('20 Newsgroups Distribucija', fontweight='bold', fontsize=14)
        
        for i, (bar, val) in enumerate(zip(bars, cat_counts.values)):
            ax.text(val + 10, i, str(val), va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Top Keywords Tabela
        st.markdown("---")
        st.markdown("**Top 5 Ključnih Besed po Kategorijah**")
        
        # Uporabimo najboljši model za ta dataset
        model_key = '20news_all_Linear_SVM'  # Najboljši model
        if model_key in data['models']:
            model_data = data['models'][model_key]
            keywords_df = get_top_keywords_for_categories(
                model_data['vectorizer'],
                model_data['model'],
                data['categories_20news'],
                n_words=5
            )
            
            if keywords_df is not None:
                st.info(f"Ključne besede pridobljene iz modela: Linear SVM")
                st.dataframe(keywords_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Ključne besede niso na voljo za ta model.")
        else:
            st.warning("Model za prikaz ključnih besed ni na voljo. Zaženite train_multi_dataset.py")
        
        # Vzorčni dokumenti
        with st.expander("Poglej Vzorčne Dokumente"):
            sample_cat = st.selectbox("Izberi kategorijo:", data['categories_20news'], key='sample_20news')
            sample_docs = data['df_20news'][data['df_20news']['category'] == sample_cat].head(3)
            
            for i, (idx, row) in enumerate(sample_docs.iterrows(), 1):
                st.markdown(f"**Vzorec {i}:**")
                st.text(row['text'][:300] + "...")
                st.markdown("---")
    
    # Zavihek 2: Širše Kategorije
    with tab2:
        st.subheader("20 Newsgroups (Širši - 6 Kategorij)")
        
        st.info("""
        **Poenostavitev kategorij**: 20 originalnih kategorij je združenih v 6 širših skupin 
        za zmanjšanje kompleksnosti naloge in demonstracijo uspešnosti pri različnih težavnostih klasifikacije.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Skupaj Dokumentov", f"{len(data['df_broader']):,}")
        with col2:
            st.metric("Kategorij", len(data['categories_broader']))
        with col3:
            st.metric("Preslikava", "20 → 6")
        
        st.markdown("---")
        
        # Prikaz preslikave
        st.markdown("**Preslikava Kategorij:**")
        
        mapping_info = broader_categories.get_broader_category_info()
        
        for info in mapping_info:
            with st.expander(f"{info['broader_category']} ({info['num_original']} kategorij)"):
                for orig_cat in info['original_categories']:
                    st.markdown(f"- {orig_cat}")
        
        st.markdown("---")
        
        # Distribucija
        st.markdown("**Distribucija Širših Kategorij**")
        
        broader_counts = data['df_broader']['broader_category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(broader_counts)), broader_counts.values, color='#4CAF50')
        ax.set_xticks(range(len(broader_counts)))
        ax.set_xticklabels(broader_counts.index, rotation=0, ha='center')
        ax.set_ylabel('Število Dokumentov', fontweight='bold')
        ax.set_title('Distribucija Širših Kategorij', fontweight='bold', fontsize=14)
        
        for bar, val in zip(bars, broader_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Top Keywords Tabela
        st.markdown("---")
        st.markdown("**Top 5 Ključnih Besed po Širših Kategorijah**")
        
        model_key = '20news_broader_Linear_SVM'
        if model_key in data['models']:
            model_data = data['models'][model_key]
            keywords_df = get_top_keywords_for_categories(
                model_data['vectorizer'],
                model_data['model'],
                data['categories_broader'],
                n_words=5
            )
            
            if keywords_df is not None:
                st.info(f"Ključne besede pridobljene iz modela: Linear SVM")
                st.dataframe(keywords_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Ključne besede niso na voljo za ta model.")
        else:
            st.warning("Model za prikaz ključnih besed ni na voljo.")
    
    # Zavihek 3: Slovenski
    with tab3:
        st.subheader("Slovenski Podatkovni Nabor")
        
        if data['df_slovenian'] is not None:
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skupaj Dokumentov", len(data['df_slovenian']))
            with col2:
                st.metric("Kategorij", len(data['categories_slovenian']))
            with col3:
                st.metric("Jezik", "Slovenščina")
            
            st.markdown("---")
            
            st.markdown("**Kategorije:**")
            for cat in data['categories_slovenian']:
                st.markdown(f"- {cat}")
            
            st.markdown("---")
            
            # Distribucija
            st.markdown("**Distribucija Slovenskih Kategorij**")
            slo_counts = data['df_slovenian']['category'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(range(len(slo_counts)), slo_counts.values, color='#FF9800')
            ax.set_xticks(range(len(slo_counts)))
            ax.set_xticklabels(slo_counts.index, rotation=0, ha='center')
            ax.set_ylabel('Število Dokumentov', fontweight='bold')
            ax.set_title('Distribucija Slovenskega Dataseta', fontweight='bold', fontsize=14)
            
            for bar, val in zip(bars, slo_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       str(val), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Top Keywords Tabela
            st.markdown("---")
            st.markdown("**Top 5 Ključnih Besed po Slovenskih Kategorijah**")
            
            model_key = 'slovenian_Linear_SVM'
            if model_key in data['models']:
                model_data = data['models'][model_key]
                keywords_df = get_top_keywords_for_categories(
                    model_data['vectorizer'],
                    model_data['model'],
                    data['categories_slovenian'],
                    n_words=5
                )
                
                if keywords_df is not None:
                    st.info(f"Ključne besede pridobljene iz modela: Linear SVM")
                    st.dataframe(keywords_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Ključne besede niso na voljo za ta model.")
            else:
                st.warning("Model za prikaz ključnih besed ni na voljo.")
            
            # Vzorci
            with st.expander("Poglej Vzorčne Dokumente"):
                sample_slo = data['df_slovenian'].sample(min(3, len(data['df_slovenian'])))
                for i, (idx, row) in enumerate(sample_slo.iterrows(), 1):
                    st.markdown(f"**{row['category']}:**")
                    st.text(row['text'][:200] + "...")
                    st.markdown("---")
        else:
            st.warning("Slovenski dataset ni najden. Zaženite slovenian_placeholder.py za ustvarjanje.")
    
    # Procesna Veriga Predprocesiranja
    st.markdown("---")
    st.subheader("Procesna Veriga Predprocesiranja")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        1. **Čiščenje Besedila**
           - Pretvorba v male črke
           - Odstranitev posebnih znakov
           - Normalizacija presledkov
           - Odstranitev dokumentov < 5 znakov
        
        2. **Razdelitev Train/Test**
           - 80% treniranje, 20% testiranje
           - Stratificirana razdelitev (uravnotežene kategorije)
        
        3. **Vektorizacija**
           - Pretvorba besedila v numerične značilke
           - TF-IDF vektorizacija (najboljša uspešnost)
        
        """)
    
    with col2:
        if len(data['df_20news']) > 0:
            sample = data['df_20news'].iloc[0]

        st.info(
            f"**Primer Pred/Po:**\n\n"
            f"**Originalno:** {sample['text'][:100]}...\n\n"
            f"**Očiščeno:** \n\n{sample['clean_text'][:100]}..."
        )



# ==========================================
# PAGE 3: VECTORIZATION
# ==========================================

def page_vectorization(data):
    """Stran 3: Vektorizacija."""
    
    st.title("Vektorizacija Besedil")
    st.markdown("Primerjava različnih metod pretvorbe besedil v numerične vektorje")
    
    st.markdown("---")
    
    # Razlaga zakaj vektorizacija
    st.subheader("Zakaj Vektorizacija?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Problem:** Algoritmi strojnega učenja ne morejo neposredno obdelati besedila - potrebujejo številke.
        
        **Rešitev:** Vektorizacija pretvori besedilo v numerične vektorje, pri čemer ohrani semantični pomen.
        
        **Tri glavne metode:**
        
        1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
           - Uteži besede glede na pomembnost
           - Zmanjša vpliv zelo pogostih besed
           - Najboljša natančnost
        
        2. **Count Vectorizer (Bag of Words)**
           - Preprosto štetje besed
           - Vsaka beseda = ena dimenzija
           - Hitra, enostavna
        
        3. **Hashing Vectorizer**
           - Uporablja hash funkcije
           - Ne potrebuje slovarja
           - Najhitrejša za velike datasete
        """)
    
    with col2:
        st.info("""
        **Primer:**
        
        Besedilo:
        "mačka sedi"
        
        Vektor:
        [0, 1, 0, 1, ...]
        
        Vsaka pozicija
        predstavlja besedo
        v slovarju.
        """)
    
    st.markdown("---")
    
    # Prikaz rezultatov primerjave
    st.subheader("Rezultati Primerjave")
    
    if data['vectorizer_results']:
        vec_results = data['vectorizer_results']
        
        # Tabela primerjave
        comparison_data = []
        for name, res in vec_results.items():
            comparison_data.append({
                'Vektorizator': name,
                'Čas Fit (s)': f"{res['fit_time']:.3f}",
                'Čas Transform (s)': f"{res['transform_time']:.3f}",
                'Skupaj (s)': f"{res['total_time']:.3f}",
                'Razpršenost': f"{res['sparsity']*100:.2f}%",
                'Točnost': f"{res['accuracy']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Vizualizacija
        if os.path.exists('data/vectorizer_comparison.png'):
            st.markdown("---")
            st.subheader("Vizualna Primerjava")
            st.image('data/vectorizer_comparison.png')
        
        st.markdown("---")
        
        # Zaključek
        st.subheader("Zaključek")
        
        best_acc = max(vec_results.items(), key=lambda x: x[1]['accuracy'])
        fastest = min(vec_results.items(), key=lambda x: x[1]['total_time'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Najboljša točnost:** {best_acc[0]}")
            st.write(f"Točnost: {best_acc[1]['accuracy']:.4f}")
        
        with col2:
            st.info(f"**Najhitrejši:** {fastest[0]}")
            st.write(f"Čas: {fastest[1]['total_time']:.3f}s")
        
        st.markdown("---")
        
        st.markdown("""
        **Odločitev:** Za vse nadaljnje modele uporabljamo **TF-IDF**, ker:
        - Dosega najvišjo natančnost
        - Standardna izbira v industriji
        - Dobro razmerje med hitrostjo in točnostjo
        - Intuitivna interpretacija (pomembnost besed)
        """)
        
    else:
        st.warning("Rezultati primerjave vektorizatorjev niso na voljo. Zaženite vectorizer_comparison.py")
        
        st.markdown("""
        **Pričakovani rezultati:**
        
        Po analizi treh metod vektorizacije:
        - TF-IDF: ~87% točnost (najboljša)
        - Count: ~86% točnost
        - Hashing: ~85% točnost (najhitrejša)
        
        **TF-IDF izbran** za vse modele zaradi najboljše uspešnosti.
        """)


# ==========================================
# PAGE 4: MODELS & OPTIMIZATION
# ==========================================

def page_models_optimization(data):
    """Stran 4: Modeli in Optimizacija."""
    
    st.title("Modeli in Optimizacija")
    st.markdown("Grid Search in primerjava treh ML algoritmov")
    
    st.markdown("---")
    
    # Razlaga Grid Search
    st.subheader("Kaj je Grid Search?")
    
    st.markdown("""
    **Grid Search** je sistematična metoda za iskanje najboljših hiperparametrov modela.
    
    **Proces:**
    1. Definiramo mrežo možnih vrednosti parametrov
    2. Treniramo model za vsako kombinacijo
    3. Evalviramo z navzkrižno validacijo (cross-validation)
    4. Izberemo kombinacijo z najboljšo uspešnostjo
    """)
    
    st.markdown("---")
    
    # Trenirani modeli
    st.subheader("Trenirani Modeli")
    
    if data['training_summary']:
        
        # Izbira modela za pregled
        model_types = ['Naive Bayes', 'Logistic Regression', 'Linear SVM']
        
        for model_type in model_types:
            with st.expander(f"**{model_type}**"):
                
                # Najdi vse datasete za ta model
                model_results = {k: v for k, v in data['training_summary'].items() 
                               if model_type.replace(' ', '_') in k}
                
                if model_results:
                    # Opis modela
                    if model_type == 'Naive Bayes':
                        st.markdown("""
                        **Naive Bayes** je probabilistični klasifikator, ki temelji na Bayesovem teoremu.
                        
                        **Prednosti:** Hiter, enostaven, dobro deluje na tekstovnih podatkih
                        
                        **Parametri:** `alpha` (Laplace smoothing)
                        """)
                    elif model_type == 'Logistic Regression':
                        st.markdown("""
                        **Logistic Regression** je linearen model za klasifikacijo.
                        
                        **Prednosti:** Dobra interpretabilnost, stabilna konvergenca
                        
                        **Parametri:** `C` (regularizacija), `solver` (optimizacijski algoritem)
                        """)
                    else:  # Linear SVM
                        st.markdown("""
                        **Linear SVM** išče optimalno hiperravnino za ločevanje razredov.
                        
                        **Prednosti:** Odlična natančnost, robusten na visoko-dimenzionalne podatke
                        
                        **Parametri:** `C` (kompromis med margino in napakami)
                        """)
                    
                    st.markdown("---")
                    
                    # Rezultati po datasetih
                    st.markdown("**Rezultati po Podatkovnih Naborih:**")
                    
                    for key, res in model_results.items():
                        dataset_name = key.split('_')[0]
                        if 'broader' in key:
                            dataset_name = '20news_broader'
                        
                        dataset_display = {
                            '20news': '20 (Vsi)',
                            '20news_broader': '20 (Širši)',
                            'slovenian': 'Slovenski'
                        }.get(dataset_name, dataset_name)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Nabor podatkov", dataset_display)
                        with col2:
                            st.metric("CV Točnost", f"{res['cv_score']:.4f}")
                        with col3:
                            st.metric("Test Točnost", f"{res['test_score']:.4f}")
                        with col4:
                            st.metric("Čas", f"{res['train_time']:.2f}s")
                        
                        # Najboljši parametri
                        st.markdown(f"**Najboljši parametri:** {res['best_params']}")
                        
                        st.markdown("---")
        
        # Skupna primerjava
        st.markdown("---")
        st.subheader("Skupna Primerjava")
        
        # Pripravi podatke za graf
        datasets_order = ['20news_all', '20news_broader', 'slovenian']
        models_order = ['Naive_Bayes', 'Logistic_Regression', 'Linear_SVM']
        
        results_matrix = []
        
        for dataset in datasets_order:
            for model in models_order:
                key = f"{dataset}_{model}"
                if key in data['training_summary']:
                    res = data['training_summary'][key]
                    results_matrix.append({
                        'Dataset': dataset.replace('_', ' ').title(),
                        'Model': model.replace('_', ' '),
                        'Test Točnost': res['test_score']
                    })
        
        if results_matrix:
            df_results = pd.DataFrame(results_matrix)
            
            # Stolpčni graf
            fig, ax = plt.subplots(figsize=(12, 6))
            
            datasets_unique = df_results['Dataset'].unique()
            models_unique = df_results['Model'].unique()
            
            x = np.arange(len(datasets_unique))
            width = 0.25
            
            for i, model in enumerate(models_unique):
                model_data = df_results[df_results['Model'] == model]
                scores = [model_data[model_data['Dataset'] == d]['Test Točnost'].values[0] 
                         if len(model_data[model_data['Dataset'] == d]) > 0 else 0 
                         for d in datasets_unique]
                
                offset = (i - 1) * width
                bars = ax.bar(x + offset, scores, width, 
                            label=model, alpha=0.8)
                
                # Dodaj vrednosti na stolpce
                for bar, score in zip(bars, scores):
                    if score > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                              f'{score:.3f}', ha='center', va='bottom', 
                              fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Nabor Podatkov', fontweight='bold', fontsize=12)
            ax.set_ylabel('Test Točnost', fontweight='bold', fontsize=12)
            ax.set_title('Primerjava Modelov po Naboru Podatkov', fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets_unique, rotation=0, ha='center')
            ax.legend()
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Najboljši modeli
            st.markdown("---")
            st.subheader("Najboljši Modeli")
            
            for dataset in datasets_unique:
                dataset_models = df_results[df_results['Dataset'] == dataset]
                best = dataset_models.loc[dataset_models['Test Točnost'].idxmax()]
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.write(f"**{dataset}**")
                with col2:
                    st.write(f"{best['Model']}")
                with col3:
                    st.write(f"**{best['Test Točnost']:.2%}**")
    
    else:
        st.warning("Rezultati treniranja niso na voljo. Zaženite train_multi_dataset.py")


# ==========================================
# PAGE 5: RESULTS EXPLORER
# ==========================================

def page_results_explorer(data):
    """Stran 5: Raziskovalec Rezultatov - Glavna stran."""
    
    st.title("Rezultati Modelov")
    st.markdown("Podrobna analiza uspešnosti modelov")
    
    st.markdown("---")
    
    # Izbira dataseta in modela
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_options = {
            '20news_all': '20 Newsgroups (Vsi - 20 kategorij)',
            '20news_broader': '20 Newsgroups (Širši - 6 kategorij)',
            'slovenian': 'Slovenski Teksti'
        }
        selected_dataset = st.selectbox(
            "Izberi Podaktovni Nabor:",
            options=list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x]
        )
    
    with col2:
        model_options = ['Naive_Bayes', 'Logistic_Regression', 'Linear_SVM']
        model_display = {
            'Naive_Bayes': 'Naive Bayes',
            'Logistic_Regression': 'Logistic Regression',
            'Linear_SVM': 'Linear SVM'
        }
        selected_model = st.selectbox(
            "Izberi Model:",
            options=model_options,
            format_func=lambda x: model_display[x]
        )
    
    # Sestavi ključ za model
    model_key = f"{selected_dataset}_{selected_model}"
    
    if model_key not in data['models']:
        st.error(f"Model {model_key} ni na voljo. Prosim zaženite train_multi_dataset.py")
        return
    
    model_data = data['models'][model_key]
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    category_names = model_data['category_names']
    
    st.markdown("---")
    
    # Metrike uspešnosti
    st.subheader("Metrike Uspešnosti")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CV Točnost", f"{model_data['cv_score']:.4f}")
    with col2:
        st.metric("Test Točnost", f"{model_data['test_score']:.4f}")
    with col3:
        st.metric("Št. Kategorij", len(category_names))
    with col4:
        st.metric("Št. Značilk", vectorizer.max_features)
    
    st.markdown(f"**Najboljši parametri:** {model_data['best_params']}")
    
    st.markdown("---")
    
    # Zavihki za različne analize
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Klasifikacijsko Poročilo", 
        "Matrika Zamenjav",
        "ROC Krivulje",
        "F1-Score po Razredih",
        "Vizualizacija v Prostoru (PCA)",
        "Primeri Klasifikacij"
    ])
    
    # Pripravi podatke za analizo
    # Naložimo dataset in naredimo napovedi
    if selected_dataset == '20news_all':
        df_analysis = data['df_20news']
        target_col = 'target'
    elif selected_dataset == '20news_broader':
        df_analysis = data['df_broader'].copy()
        df_analysis['target'] = df_analysis['broader_target']
        df_analysis['category'] = df_analysis['broader_category']
        target_col = 'target'
    else:  # slovenian
        df_analysis = data['df_slovenian']
        target_col = 'target'
    
    # Train/test split (isti kot pri treniranju)
    from sklearn.model_selection import train_test_split
    
    X_text = df_analysis['clean_text']
    y = df_analysis[target_col]
    
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize test set
    X_test = vectorizer.transform(X_text_test)
    y_pred = model.predict(X_test)
    
    # TAB 1: Classification Report
    with tab1:
        st.markdown("**Klasifikacijsko Poročilo**")
        st.markdown("Podrobne metrike za vsako kategorijo")
        
        from sklearn.metrics import classification_report
        
        report = classification_report(
            y_test, y_pred,
            target_names=category_names,
            output_dict=True,
            zero_division=0
        )
        
        # Pretvori v DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Filtriraj samo kategorije (ne accuracy, macro avg, itd.)
        category_report = report_df[report_df.index.isin(category_names)]
        
        # Formatiraj
        category_report = category_report[['precision', 'recall', 'f1-score', 'support']]
        category_report.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
        
        st.dataframe(
            category_report.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'Support': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Povzetne metrike
        st.markdown("---")
        st.markdown("**Povzetne Metrike:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Macro Avg Precision", f"{report['macro avg']['precision']:.3f}")
        with col2:
            st.metric("Macro Avg Recall", f"{report['macro avg']['recall']:.3f}")
        with col3:
            st.metric("Macro Avg F1-Score", f"{report['macro avg']['f1-score']:.3f}")
    
    # TAB 2: Confusion Matrix
    with tab2:
        st.markdown("**Matrika Zamenjav**")
        st.markdown("Vizualizacija napak klasifikacije")
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Če je preveč kategorij, uporabi manjši font
        if len(category_names) > 10:
            fig_size = (14, 12)
            font_size = 8
        else:
            fig_size = (10, 8)
            font_size = 10
        
        fig, ax = plt.subplots(figsize=fig_size)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='OrRd',
            xticklabels=category_names,
            yticklabels=category_names,
            cbar_kws={'label': 'Število'},
            ax=ax
        )
        
        ax.set_xlabel('Napovedano', fontweight='bold', fontsize=12)
        ax.set_ylabel('Dejansko', fontweight='bold', fontsize=12)
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
        
        plt.xticks(rotation=45, ha='center', fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # TAB 3: ROC Krivulje
    with tab3:
        st.markdown("**ROC Krivulje (Receiver Operating Characteristic)**")
        st.markdown("Analiza kompromisa med True Positive Rate in False Positive Rate")
        
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        # Preveri če model podpira verjetnosti
        if hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        elif hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
        else:
            st.warning("Ta model ne podpira verjetnostnih napovedi, zato ROC krivulj ni mogoče izračunati.")
            st.info("ROC krivulje so na voljo za: Linear SVM in Logistic Regression")
        
        if hasattr(model, 'decision_function') or hasattr(model, 'predict_proba'):
            # Binarizacija oznak za One-vs-Rest
            n_classes = len(category_names)
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            # Izračunaj ROC za vsak razred
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Micro-average
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # Macro-average
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
            # Prikaz metrik
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Micro-Average AUC", f"{roc_auc['micro']:.4f}")
            with col2:
                st.metric("Macro-Average AUC", f"{roc_auc['macro']:.4f}")
            
            st.markdown("---")
            
            # Vizualizacija
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Graf 1: Micro in Macro average
            ax = axes[0]
            ax.plot(fpr["micro"], tpr["micro"],
                   label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                   color='deeppink', linestyle=':', linewidth=3)
            ax.plot(fpr["macro"], tpr["macro"],
                   label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                   color='navy', linestyle=':', linewidth=3)
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Naključni klasifikator')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
            ax.set_title('ROC Krivulja - Povprečja', fontweight='bold', fontsize=12)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(alpha=0.3)
            
            # Graf 2: Top 5 razredov po AUC
            ax = axes[1]
            
            # Sortiraj razrede po AUC
            class_auc = [(i, roc_auc[i]) for i in range(n_classes)]
            class_auc_sorted = sorted(class_auc, key=lambda x: x[1], reverse=True)[:5]
            
            colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            
            for (class_idx, auc_score), color in zip(class_auc_sorted, colors):
                ax.plot(fpr[class_idx], tpr[class_idx], color=color, linewidth=2,
                       label=f'{category_names[class_idx][:20]} (AUC={auc_score:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Naključni')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
            ax.set_title('ROC Krivulje - Top 5 Razredov', fontweight='bold', fontsize=12)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabela z AUC za vse razrede
            with st.expander("Poglej AUC za vse razrede"):
                auc_data = []
                for i in range(n_classes):
                    auc_data.append({
                        'Kategorija': category_names[i],
                        'AUC': roc_auc[i],
                        'Vzorci': int((y_test_bin[:, i] == 1).sum())
                    })
                
                auc_df = pd.DataFrame(auc_data).sort_values('AUC', ascending=False)
                
                st.dataframe(
                    auc_df.style.format({
                        'AUC': '{:.4f}',
                        'Vzorci': '{:d}'
                    }),
                    use_container_width=True
                )
    
    # TAB 4: F1-Score po Razredih
    with tab4:
        st.markdown("**F1-Score po Razredih**")
        st.markdown("Vizualizacija uspešnosti za vsako kategorijo")
        
        # Pridobi F1-scores iz classification report
        from sklearn.metrics import classification_report
        
        report = classification_report(
            y_test, y_pred,
            target_names=category_names,
            output_dict=True,
            zero_division=0
        )
        
        # Ekstrahiraj F1-scores
        f1_scores = [report[cat]['f1-score'] for cat in category_names]
        precisions = [report[cat]['precision'] for cat in category_names]
        recalls = [report[cat]['recall'] for cat in category_names]
        supports = [report[cat]['support'] for cat in category_names]
        
        # Sortiraj po F1-score
        sorted_indices = np.argsort(f1_scores)[::-1]
        
        sorted_categories = [category_names[i] for i in sorted_indices]
        sorted_f1 = [f1_scores[i] for i in sorted_indices]
        sorted_precision = [precisions[i] for i in sorted_indices]
        sorted_recall = [recalls[i] for i in sorted_indices]
        sorted_support = [supports[i] for i in sorted_indices]
        
        # F1-Score bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(category_names) * 0.3)))
        
        # Barve glede na uspešnost
        colors = ['#4CAF50' if f1 >= 0.8 else '#FF9800' if f1 >= 0.6 else '#F44336' 
                  for f1 in sorted_f1]
        
        bars = ax.barh(range(len(sorted_categories)), sorted_f1, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_categories)))
        ax.set_yticklabels(sorted_categories, fontsize=9)
        ax.set_xlabel('F1-Score', fontweight='bold', fontsize=11)
        ax.set_title('F1-Score po Kategorijah', fontweight='bold', fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Dodaj vrednosti
        for i, (bar, val, sup) in enumerate(zip(bars, sorted_f1, sorted_support)):
            ax.text(val + 0.01, i, f'{val:.3f} (n={int(sup)})', 
                   va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistika
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Najboljša F1", f"{max(f1_scores):.3f}")
            best_cat = category_names[np.argmax(f1_scores)]
            st.caption(f"{best_cat}")
        
        with col2:
            st.metric("Najslabša F1", f"{min(f1_scores):.3f}")
            worst_cat = category_names[np.argmin(f1_scores)]
            st.caption(f"{worst_cat}")
        
        with col3:
            st.metric("Povprečna F1", f"{np.mean(f1_scores):.3f}")
            st.caption("Macro average")
        
        with col4:
            st.metric("Std. Odklon", f"{np.std(f1_scores):.3f}")
            st.caption("Konsistentnost")
    
    
    # TAB 5: PCA Visualization
    with tab5:
        st.markdown("**Vizualizacija v Prostoru (PCA)**")
        st.markdown("2D projekcija dokumentov z Principal Component Analysis")
        
        st.info("""
        **PCA (Principal Component Analysis)** zmanjša dimenzionalnost iz 5000 značilk na 2 dimenziji,
        pri čemer ohrani čim več variance. To nam omogoča vizualizacijo kako se kategorije ločujejo v prostoru.
        """)
        
        # Vzorčenje za hitrost (če je preveč dokumentov)
        max_samples = 2000
        n_samples = X_test.shape[0]
        if n_samples > max_samples:
            sample_indices = np.random.choice(n_samples, max_samples, replace=False)
            X_sample = X_test[sample_indices]
            y_sample = y_test.iloc[sample_indices].values
        else:
            X_sample = X_test
            y_sample = y_test.values
        
        with st.spinner("Računam PCA projekcijo..."):
            from sklearn.decomposition import PCA
            
            # PCA na 2 komponenti
            pca = PCA(n_components=2, random_state=42)
            
            # Če je sparse matrix, pretvori v dense (samo za sample)
            if hasattr(X_sample, 'toarray'):
                X_dense = X_sample.toarray()
            else:
                X_dense = X_sample
            
            X_pca = pca.fit_transform(X_dense)
            
            # Pripravi barve za kategorije
            n_categories = len(category_names)
            
            if n_categories <= 10:
                # Uporabi kvalitetne barve za <= 10 kategorij
                colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
            else:
                # Za več kategorij uporabi gradient
                colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(14, 10))
            
            for i, category in enumerate(category_names):
                mask = y_sample == i
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[colors[i]], label=category, 
                          alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                         fontweight='bold', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                         fontweight='bold', fontsize=12)
            ax.set_title('PCA Vizualizacija Dokumentov po Kategorijah', 
                        fontweight='bold', fontsize=14)
            
            # Legenda
            if n_categories <= 12:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            else:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
            
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistika
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Skupna razložena varianca", 
                         f"{(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.1f}%")
            with col2:
                st.metric("PC1 varianca", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
            with col3:
                st.metric("PC2 varianca", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
            
            st.markdown("""
            **Interpretacija:**
            - Bližje kot so točke iste barve, bolj podobni so dokumenti te kategorije
            - Če so kategorije dobro ločene (gručaste), jih model lahko dobro razlikuje
            - Prekrivanje med kategorijami kaže na težave pri klasifikaciji
            - PCA ohranja globalne strukture podatkov
            """)
    
    # TAB 6: Primeri
    with tab6:
        st.markdown("**Primeri Klasifikacij**")
        
        # Filter za prikaz
        show_filter = st.radio(
            "Prikaži:",
            ["Vsi", "Samo pravilne", "Samo napačne"],
            horizontal=True
        )
        
        num_examples = st.slider("Število primerov:", 5, 50, 10)
        
        # Sestavi DataFrame z rezultati
        results_df = pd.DataFrame({
            'text': X_text_test.values,
            'true_label': y_test.values,
            'pred_label': y_pred,
            'true_category': [category_names[i] for i in y_test.values],
            'pred_category': [category_names[i] for i in y_pred],
            'correct': y_test.values == y_pred
        })
        
        # Filtriraj
        if show_filter == "Samo pravilne":
            results_df = results_df[results_df['correct'] == True]
        elif show_filter == "Samo napačne":
            results_df = results_df[results_df['correct'] == False]
        
        # Vzorči
        display_df = results_df.head(num_examples)
        
        if len(display_df) == 0:
            st.info("Ni primerov za prikaz s trenutnimi filtri.")
        else:
            for idx, row in display_df.iterrows():
                if row['correct']:
                    st.success(f"**Pravilno klasificirano**")
                else:
                    st.error(f"**Napačno klasificirano**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Dejanska kategorija:** {row['true_category']}")
                with col2:
                    st.write(f"**Napovedana kategorija:** {row['pred_category']}")
                
                # Prikaz besedila
                text_preview = row['text'][:300] + ("..." if len(row['text']) > 300 else "")
                st.text(text_preview)
                
                st.markdown("---")


# ==========================================
# PAGE 6: COMPARISONS
# ==========================================

def page_comparisons(data):
    """Stran 6: Primerjave Modelov in Naborov."""
    
    st.title("Primerjave")
    st.markdown("Analiza uspešnosti različnih modelov in podatkovnih naborov")
    
    st.markdown("---")
    
    # Dva dela
    st.subheader("A) Primerjava Podatkovnih Naborov")
    st.markdown("Kako se isti model obnaša na različnih naborih podatkov?")
    
    # Izbira modela
    model_options = ['Naive_Bayes', 'Logistic_Regression', 'Linear_SVM']
    model_display = {
        'Naive_Bayes': 'Naive Bayes',
        'Logistic_Regression': 'Logistic Regression',
        'Linear_SVM': 'Linear SVM'
    }
    
    selected_model_comp = st.selectbox(
        "Izberi Model za Primerjavo:",
        options=model_options,
        format_func=lambda x: model_display[x],
        key='dataset_comp_model'
    )
    
    # Pridobi rezultate za ta model na vseh datasetih
    datasets = ['20news_all', '20news_broader', 'slovenian']
    dataset_display = {
        '20news_all': '20 Newsgroups (Vsi)',
        '20news_broader': '20 Newsgroups (Širši)',
        'slovenian': 'Slovenski'
    }
    
    dataset_results = []
    for dataset in datasets:
        key = f"{dataset}_{selected_model_comp}"
        if key in data['training_summary']:
            dataset_results.append({
                'Dataset': dataset_display[dataset],
                'CV Točnost': data['training_summary'][key]['cv_score'],
                'Test Točnost': data['training_summary'][key]['test_score'],
                'Št. Kategorij': len(data['models'][key]['category_names']) if key in data['models'] else 0
            })
    
    if dataset_results:
        # Graf
        df_datasets = pd.DataFrame(dataset_results)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df_datasets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_datasets['CV Točnost'], width, 
                      label='CV Točnost', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, df_datasets['Test Točnost'], width, 
                      label='Test Točnost', color='#4CAF50', alpha=0.8)
        
        ax.set_xlabel('Podatkovni Nabor', fontweight='bold', fontsize=12)
        ax.set_ylabel('Točnost', fontweight='bold', fontsize=12)
        ax.set_title(f'Primerjava Podatkovnih Naborov - {model_display[selected_model_comp]}', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df_datasets['Dataset'], rotation=15, ha='center')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Dodaj vrednosti
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabela
        st.dataframe(df_datasets, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Opazovanja:**
        - Širše kategorije (6) so lažje za klasifikacijo kot vse kategorije (20)
        - Slovenski dataset je manjši in ima drugačen jezik (slabša točnost pričakovana)
        - Model je treniran in optimiziran posebej za vsak dataset
        """)
    
    else:
        st.warning("Podatki za primerjavo niso na voljo.")
    
    # Del B: Primerjava Modelov
    st.markdown("---")
    st.subheader("B) Primerjava Modelov")
    st.markdown("Kateri model je najboljši na določenem naboru podatkov?")
    
    # Izbira dataseta
    dataset_comp_options = {
        '20news_all': '20 Newsgroups (Vsi - 20 kategorij)',
        '20news_broader': '20 Newsgroups (Širši - 6 kategorij)',
        'slovenian': 'Slovenski Teksti'
    }
    
    selected_dataset_comp = st.selectbox(
        "Izberi Podatkovni Nabor za Primerjavo:",
        options=list(dataset_comp_options.keys()),
        format_func=lambda x: dataset_comp_options[x],
        key='model_comp_dataset'
    )
    
    # Pridobi rezultate za vse modele na tem datasetu
    model_results = []
    for model in model_options:
        key = f"{selected_dataset_comp}_{model}"
        if key in data['training_summary']:
            model_results.append({
                'Model': model_display[model],
                'CV Točnost': data['training_summary'][key]['cv_score'],
                'Test Točnost': data['training_summary'][key]['test_score'],
                'Čas (s)': data['training_summary'][key].get('train_time', 0)
            })
    
    if model_results:
        # Graf
        df_models = pd.DataFrame(model_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Graf 1: Točnost
        ax = axes[0]
        
        x = np.arange(len(df_models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_models['CV Točnost'], width, 
                      label='CV Točnost', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, df_models['Test Točnost'], width, 
                      label='Test Točnost', color='#4CAF50', alpha=0.8)
        
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_ylabel('Točnost', fontweight='bold', fontsize=12)
        ax.set_title('Primerjava Točnosti Modelov', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df_models['Model'], rotation=15, ha='center')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Graf 2: Čas treniranja
        ax = axes[1]
        
        bars = ax.barh(df_models['Model'], df_models['Čas (s)'], color='#FF9800', alpha=0.8)
        ax.set_xlabel('Čas Treniranja (s)', fontweight='bold', fontsize=12)
        ax.set_title('Primerjava Hitrosti', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, df_models['Čas (s)'])):
            ax.text(val + 1, i, f'{val:.1f}s', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabela
        st.dataframe(df_models, use_container_width=True, hide_index=True)
        
        # Najboljši model
        best_model = df_models.loc[df_models['Test Točnost'].idxmax()]
        fastest_model = df_models.loc[df_models['Čas (s)'].idxmin()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Najboljša točnost:** {best_model['Model']}")
            st.write(f"Test točnost: {best_model['Test Točnost']:.4f}")
        
        with col2:
            st.info(f"**Najhitrejši:** {fastest_model['Model']}")
            st.write(f"Čas: {fastest_model['Čas (s)']:.3f}s")
        
        st.markdown("""
        **Opazovanja:**
        - Linear SVM običajno dosega najboljšo točnost
        - Naive Bayes je najhitrejši, a manj natančen
        - Logistic Regression ponuja dobro ravnotežje med hitrostjo in točnostjo
        """)
    
    else:
        st.warning("Podatki za primerjavo niso na voljo.")

# ==========================================
# PAGE 7: LIVE PREDICTION
# ==========================================

def page_live_prediction(data):
    """Stran 7: Interaktivna Napoved."""
    
    st.title("Interaktivna Napoved")
    st.markdown("Vnesite besedilo in preizkusite klasifikacijo")
    
    st.markdown("---")
    
    # Izbira modela
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_options = {
            '20news_all': '20 Newsgroups (20 kategorij)',
            '20news_broader': '20 Newsgroups (6 kategorij)',
            'slovenian': 'Slovenski Teksti'
        }
        selected_dataset = st.selectbox(
            "Izberi Nabor Podatkov:",
            options=list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x],
            key='pred_dataset'
        )
    
    with col2:
        model_options = ['Naive_Bayes', 'Logistic_Regression', 'Linear_SVM']
        model_display = {
            'Naive_Bayes': 'Naive Bayes',
            'Logistic_Regression': 'Logistic Regression',
            'Linear_SVM': 'Linear SVM'
        }
        selected_model = st.selectbox(
            "Izberi Model:",
            options=model_options,
            format_func=lambda x: model_display[x],
            key='pred_model'
        )
    
    # Sestavi ključ
    model_key = f"{selected_dataset}_{selected_model}"
    
    if model_key not in data['models']:
        st.error(f"Model {model_key} ni na voljo.")
        return
    
    model_data = data['models'][model_key]
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    category_names = model_data['category_names']
    
    st.markdown("---")
    
    # Vnos besedila
    user_text = st.text_area(
        "Vnesite besedilo za klasifikacijo:",
        height=200,
        placeholder="Primer: The new smartphone features an amazing camera and long battery life..."
    )
    
    if st.button("Klasificiraj", type="primary"):
        if not user_text.strip():
            st.warning("Prosim vnesite besedilo!")
        else:
            # Očisti besedilo
            cleaned = utils.clean_text(user_text)
            
            if not cleaned:
                st.error("Besedilo je prazno po čiščenju.")
            else:
                with st.spinner("Analiziram..."):
                    # Vektoriziraj
                    X_user = vectorizer.transform([cleaned])
                    
                    # Napoved
                    prediction = model.predict(X_user)[0]
                    predicted_category = category_names[prediction]
                    
                    # Verjetnosti (če model podpira)
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_user)[0]
                    elif hasattr(model, 'decision_function'):
                        decision_scores = model.decision_function(X_user)[0]
                        # Pretvori v verjetnosti
                        exp_scores = np.exp(decision_scores - decision_scores.max())
                        probabilities = exp_scores / exp_scores.sum()
                    else:
                        probabilities = None
                
                st.success("Napoved končana!")
                st.markdown("---")
                
                # Rezultat
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Napovedana Kategorija:")
                    st.markdown(f"# {predicted_category}")
                
                with col2:
                    if probabilities is not None:
                        confidence = probabilities[prediction]
                        st.metric("Zaupanje", f"{confidence*100:.1f}%")
                        
                        if confidence > 0.8:
                            st.success("Visoko zaupanje")
                        elif confidence > 0.5:
                            st.warning("Srednje zaupanje")
                        else:
                            st.error("Nizko zaupanje")
                
                # Top 5 napovedi
                if probabilities is not None:
                    st.markdown("---")
                    st.subheader("Top 5 Napovedi:")
                    
                    top5_idx = np.argsort(probabilities)[-5:][::-1]
                    
                    for i, idx in enumerate(top5_idx):
                        prob = probabilities[idx]
                        cat = category_names[idx]
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.progress(prob)
                        with col2:
                            st.write(f"**{prob*100:.1f}%**")
                        
                        st.write(f"{i+1}. {cat}")
                        
                        if i < 4:
                            st.write("")


# ==========================================
# MAIN APP
# ==========================================

def main():
    """Glavna aplikacija."""
    
    # Naloži podatke
    data = load_all_data()
    
    if data is None:
        st.error("Napaka pri nalaganju podatkov. Preverite datoteke.")
        st.stop()
    
    # Stranska vrstica - navigacija
    with st.sidebar:
        st.title("Navigacija")
        
        page = st.radio(
            "Izberi stran:",
            [
                "Pregled Projekta",
                "Podatki in Predprocesiranje",
                "Vektorizacija",
                "Modeli in Optimizacija",
                "Rezultati Modelov",
                "Primerjave",
                "Interaktivna Napoved"
            ]
        )
        
    
    # Usmerjanje na strani
    if page == "Pregled Projekta":
        page_project_overview(data)
    
    elif page == "Podatki in Predprocesiranje":
        page_data_preprocessing(data)
    
    elif page == "Vektorizacija":
        page_vectorization(data)
    
    elif page == "Modeli in Optimizacija":
        page_models_optimization(data)
    
    elif page == "Rezultati Modelov":
        page_results_explorer(data)

    elif page == "Primerjave":
        page_comparisons(data)
    
    elif page == "Interaktivna Napoved":
        page_live_prediction(data)


if __name__ == "__main__":
    main()
