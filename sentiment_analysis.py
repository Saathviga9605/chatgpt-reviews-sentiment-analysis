import pandas as pd
import numpy as np
import nltk
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, classification_report, roc_auc_score,
                            roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Configuration
class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MIN_REVIEW_LENGTH = 10
    MAX_TFIDF_FEATURES = 1000
    NGRAM_RANGE = (1, 2)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_nltk():
    """Download required NLTK resources."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download {resource}: {e}")
    logger.info("NLTK resources initialized")

@st.cache_data
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and perform initial data cleaning."""
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Dataset loaded: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading dataset: {e}")
        return None

class TextPreprocessor:
    """Enhanced text preprocessing for sentiment analysis."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Keep sentiment-bearing words
        base_stopwords = set(stopwords.words('english'))
        sentiment_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                          'very', 'really', 'quite', 'too', 'good', 'bad', 
                          'best', 'worst', 'great', 'terrible', 'excellent'}
        self.stop_words = base_stopwords - sentiment_words
    
    def preprocess(self, text: str) -> str:
        """Clean and preprocess text."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(tokens)

def map_rating_to_sentiment(rating: int) -> str:
    """Map numeric ratings to sentiment categories."""
    try:
        rating = int(rating)
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'
    except:
        return 'Neutral'

@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Complete data preprocessing pipeline."""
    df = df.copy()
    
    # Handle missing values
    df['review'] = df['review'].fillna('')
    df['date'] = df['date'].replace('########', np.nan)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Calculate review length if not present
    if 'review_length' not in df.columns:
        df['review_length'] = df['review'].astype(str).apply(len)
    
    # Map sentiment
    df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)
    
    # Text preprocessing
    preprocessor = TextPreprocessor()
    df['processed_review'] = df['review'].apply(preprocessor.preprocess)
    
    # Remove invalid entries
    df = df[df['processed_review'].str.len() >= Config.MIN_REVIEW_LENGTH]
    
    # Map to numeric labels
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label'] = df['sentiment'].map(sentiment_map)
    
    logger.info(f"Preprocessed dataset: {len(df)} samples")
    logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    return df

@st.cache_resource
def train_ensemble_model(df: pd.DataFrame) -> Tuple:
    """Train ensemble of ML models for sentiment classification."""
    
    try:
        texts = df['processed_review'].tolist()
        labels = df['label'].tolist()
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=Config.MAX_TFIDF_FEATURES,
            ngram_range=Config.NGRAM_RANGE,
            min_df=2,
            max_df=0.8
        )
        X = vectorizer.fit_transform(texts)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=labels
        )
        
        logger.info(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
        
        # Initialize models
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        lr_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            C=1.0
        )
        
        nb_model = MultinomialNB(alpha=0.1)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=Config.RANDOM_STATE,
            eval_metric='mlogloss'
        )
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('lr', lr_model),
                ('nb', nb_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )
        
        # Train models
        logger.info("Training ensemble model...")
        ensemble.fit(X_train, y_train)
        
        # Predictions
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        
        # ROC-AUC
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=['Negative', 'Neutral', 'Positive'],
            zero_division=0
        )
        
        logger.info(f"Model Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"\n{report}")
        
        return (ensemble, vectorizer), accuracy, precision, recall, f1, roc_auc, cm, y_test, y_pred_proba
        
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        st.error(f"Model training failed: {e}")
        return None, 0, 0, 0, 0, 0, np.zeros((3, 3)), [], []

@st.cache_data
def predict_sentiment(_models, text: str) -> Tuple[str, Dict[str, float]]:
    """Predict sentiment for new text."""
    
    if _models is None:
        return "Unknown", {}
    
    try:
        model, vectorizer = _models
        
        # Preprocess
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)
        
        if not processed_text:
            return "Neutral", {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33}
        
        # Transform and predict
        X = vectorizer.transform([processed_text])
        probs = model.predict_proba(X)[0]
        pred = np.argmax(probs)
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map[pred]
        
        confidence_scores = {
            'Negative': float(probs[0]),
            'Neutral': float(probs[1]),
            'Positive': float(probs[2])
        }
        
        return sentiment, confidence_scores
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error", {}

@st.cache_data
def generate_word_clouds(df: pd.DataFrame) -> Tuple:
    """Generate word clouds for different sentiments."""
    
    positive_text = ' '.join(df[df['sentiment'] == 'Positive']['processed_review'].dropna())
    negative_text = ' '.join(df[df['sentiment'] == 'Negative']['processed_review'].dropna())
    neutral_text = ' '.join(df[df['sentiment'] == 'Neutral']['processed_review'].dropna())
    
    pos_wc = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Greens', max_words=100).generate(positive_text or 'no data')
    
    neg_wc = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Reds', max_words=100).generate(negative_text or 'no data')
    
    neu_wc = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Blues', max_words=100).generate(neutral_text or 'no data')
    
    return pos_wc, neg_wc, neu_wc

def create_streamlit_dashboard(df, models, accuracy, precision, recall, f1, roc_auc, cm, 
                               y_test, y_pred_proba, pos_wc, neg_wc, neu_wc):
    """Create comprehensive Streamlit dashboard."""
    
    st.set_page_config(
        page_title="ChatGPT Sentiment Analysis",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 42px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 20px;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🤖 AI-Powered Sentiment Analysis for ChatGPT Reviews</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/chatgpt.png", width=100)
        st.title("📋 Navigation")
        
        page = st.radio(
            "Select Analysis View:",
            [
                "📊 Project Overview",
                "📈 EDA & Visualizations",
                "🎯 Model Performance",
                "🔍 Sentiment Prediction",
                "💡 Key Insights"
            ]
        )
        
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        st.metric("Total Reviews", len(df))
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        st.metric("F1-Score", f"{f1:.2%}")
        st.metric("ROC-AUC", f"{roc_auc:.2%}")
        
        st.markdown("---")
        st.info("**Project by:** GUVI Data Science Capstone")
    
    # Page 1: Project Overview
    if page == "📊 Project Overview":
        st.header("📊 Project Overview & Business Context")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Problem Statement")
            st.write("""
            This project analyzes ChatGPT user reviews to classify sentiment as 
            **Positive**, **Neutral**, or **Negative**. The goal is to understand 
            customer satisfaction and identify areas for improvement.
            """)
            
            st.subheader("💼 Business Use Cases")
            st.write("""
            1. **Customer Feedback Analysis** - Improve product features
            2. **Brand Reputation Management** - Monitor sentiment trends
            3. **Feature Enhancement** - Identify improvement areas
            4. **Automated Support** - Prioritize complaints
            5. **Marketing Optimization** - Develop engagement strategies
            """)
        
        with col2:
            st.subheader("🔧 Technical Approach")
            st.write("""
            **Data Preprocessing:**
            - Text cleaning & normalization
            - Stopword removal & lemmatization
            - Handling missing values
            
            **Model Training:**
            - TF-IDF feature extraction
            - Ensemble: RandomForest + LogisticRegression + NaiveBayes + XGBoost
            - Class balancing with weights
            
            **Evaluation:**
            - Accuracy, Precision, Recall, F1-Score
            - Confusion Matrix, ROC-AUC curves
            """)
            
            st.subheader("📦 Deliverables")
            st.write("""
            ✅ Cleaned & preprocessed dataset
            ✅ EDA with comprehensive visualizations
            ✅ Trained ML ensemble model
            ✅ Interactive web dashboard (Streamlit)
            ✅ Model performance report
            """)
        
        # Dataset Overview
        st.markdown("---")
        st.subheader("📂 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", len(df))
        col2.metric("Positive", len(df[df['sentiment'] == 'Positive']))
        col3.metric("Neutral", len(df[df['sentiment'] == 'Neutral']))
        col4.metric("Negative", len(df[df['sentiment'] == 'Negative']))
        
        # Sentiment Distribution
        st.subheader("🎨 Overall Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker=dict(colors=['#00CC96', '#FFA15A', '#EF553B'])
            )
        ])
        fig.update_layout(title="Sentiment Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample Data
        st.subheader("📋 Sample Reviews")
        st.dataframe(df[['date', 'rating', 'sentiment', 'review', 'platform', 'location']].head(10))
    
    # Page 2: EDA & Visualizations
    elif page == "📈 EDA & Visualizations":
        st.header("📈 Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Q1-Q3: Ratings & Keywords",
            "📅 Q4-Q6: Time & Platform",
            "👥 Q7-Q9: Users & Reviews",
            "🔤 Q10: Version Analysis"
        ])
        
        with tab1:
            # Q1: Rating Distribution
            st.subheader("Q1: Distribution of Review Ratings")
            fig = px.histogram(df, x='rating', nbins=5, 
                             title='Distribution of Review Ratings (1-5 Stars)',
                             color_discrete_sequence=['#636EFA'])
            fig.update_layout(xaxis_title="Rating", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Insight:** Shows overall user satisfaction distribution")
            
            # Q2: Helpful Reviews
            st.subheader("Q2: Reviews Marked as Helpful (>10 votes)")
            helpful_df = df[df['helpful_votes'] > 10]
            st.metric("Helpful Reviews", len(helpful_df))
            
            if len(helpful_df) > 0:
                fig = px.pie(helpful_df, names='sentiment', 
                           title='Sentiment of Helpful Reviews',
                           color_discrete_sequence=['#00CC96', '#FFA15A', '#EF553B'])
                st.plotly_chart(fig, use_container_width=True)
            st.info("**Insight:** Reviews users found most valuable")
            
            # Q3: Word Clouds
            st.subheader("Q3: Common Keywords in Reviews")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Positive Reviews**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(pos_wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Neutral Reviews**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(neu_wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            with col3:
                st.markdown("**Negative Reviews**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(neg_wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            st.info("**Insight:** Discover what users love vs. complain about")
        
        with tab2:
            # Q4: Rating over time
            st.subheader("Q4: Average Rating Over Time")
            if 'date' in df.columns and df['date'].notna().any():
                df_temp = df.dropna(subset=['date']).copy()
                df_temp['month'] = df_temp['date'].dt.to_period('M').astype(str)
                monthly_avg = df_temp.groupby('month')['rating'].mean().reset_index()
                
                fig = px.line(monthly_avg, x='month', y='rating',
                            title='Average Rating Trend Over Time',
                            markers=True, color_discrete_sequence=['#FF6692'])
                fig.update_layout(xaxis_title="Month", yaxis_title="Average Rating")
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Insight:** Track user satisfaction trends over time")
            else:
                st.warning("Date information not available")
            
            # Q5: Ratings by location
            st.subheader("Q5: Ratings by User Location")
            if 'location' in df.columns:
                location_avg = df.groupby('location')['rating'].mean().reset_index()
                location_avg = location_avg.sort_values('rating', ascending=False).head(15)
                
                fig = px.bar(location_avg, x='location', y='rating',
                           title='Average Rating by Top 15 Locations',
                           color='rating', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Insight:** Regional differences in satisfaction")
            
            # Q6: Platform comparison
            st.subheader("Q6: Web vs Mobile Platform Comparison")
            if 'platform' in df.columns:
                platform_avg = df.groupby('platform')['rating'].mean().reset_index()
                
                fig = px.bar(platform_avg, x='platform', y='rating',
                           title='Average Rating by Platform',
                           color='platform', color_discrete_sequence=['#AB63FA', '#FFA15A'])
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Insight:** Platform-specific user experience")
        
        with tab3:
            # Q7: Verified vs non-verified
            st.subheader("Q7: Verified vs Non-Verified Users")
            if 'verified_purchase' in df.columns:
                verified_avg = df.groupby('verified_purchase')['rating'].mean().reset_index()
                
                fig = px.bar(verified_avg, x='verified_purchase', y='rating',
                           title='Average Rating by Verification Status',
                           color='verified_purchase',
                           color_discrete_sequence=['#00CC96', '#EF553B'])
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Insight:** Are paying customers more satisfied?")
            
            # Q8: Review length by rating
            st.subheader("Q8: Review Length by Rating Category")
            avg_length = df.groupby('rating')['review_length'].mean().reset_index()
            
            fig = px.bar(avg_length, x='rating', y='review_length',
                       title='Average Review Length by Rating',
                       color='review_length', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Insight:** Do people write longer reviews when unhappy?")
            
            # Q9: Top words in 1-star reviews
            st.subheader("Q9: Most Mentioned Words in 1-Star Reviews")
            one_star = df[df['rating'] == 1]['processed_review'].dropna()
            if len(one_star) > 0:
                words = ' '.join(one_star).split()
                word_freq = Counter(words).most_common(15)
                word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                
                fig = px.bar(word_df, x='Word', y='Frequency',
                           title='Top 15 Words in Negative Reviews',
                           color='Frequency', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Insight:** Recurring complaints and pain points")
        
        with tab4:
            # Q10: Version ratings
            st.subheader("Q10: Ratings by ChatGPT Version")
            if 'version' in df.columns:
                version_avg = df.groupby('version')['rating'].mean().reset_index()
                version_avg = version_avg.sort_values('rating', ascending=False)
                
                fig = px.bar(version_avg, x='version', y='rating',
                           title='Average Rating by ChatGPT Version',
                           color='rating', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Insight:** Version performance comparison")
                
                # Sentiment by version
                st.subheader("Sentiment Distribution by Version")
                version_sentiment = df.groupby(['version', 'sentiment']).size().reset_index(name='count')
                
                fig = px.bar(version_sentiment, x='version', y='count', color='sentiment',
                           title='Sentiment Distribution Across Versions',
                           barmode='group',
                           color_discrete_map={'Positive': '#00CC96', 
                                             'Neutral': '#FFA15A', 
                                             'Negative': '#EF553B'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Page 3: Model Performance
    elif page == "🎯 Model Performance":
        st.header("🎯 Model Performance & Evaluation Metrics")
        
        # Performance Metrics
        st.subheader("📊 Overall Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Accuracy", f"{accuracy:.2%}", help="Overall correctness")
        col2.metric("Precision", f"{precision:.2%}", help="Positive prediction accuracy")
        col3.metric("Recall", f"{recall:.2%}", help="True positive rate")
        col4.metric("F1-Score", f"{f1:.2%}", help="Harmonic mean of precision & recall")
        col5.metric("ROC-AUC", f"{roc_auc:.2%}", help="Area under ROC curve")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Neutral', 'Positive'],
                       yticklabels=['Negative', 'Neutral', 'Positive'],
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Per-Class Performance")
            
            # Calculate per-class metrics
            for i, sentiment in enumerate(['Negative', 'Neutral', 'Positive']):
                true_pos = cm[i, i]
                total = cm[i].sum()
                class_acc = true_pos / total if total > 0 else 0
                
                st.metric(f"{sentiment} Accuracy", f"{class_acc:.2%}")
                
                # Show percentage bar
                st.progress(class_acc, text=f"{sentiment}: {class_acc:.1%}")
            
            st.markdown("---")
            st.info("""
            **Interpretation:**
            - **Negative**: Model's ability to identify negative reviews
            - **Neutral**: Model's ability to identify neutral reviews
            - **Positive**: Model's ability to identify positive reviews
            """)
        
        # ROC Curves
        st.subheader("📊 ROC-AUC Curves (Multi-Class)")
        
        if len(y_test) > 0 and len(y_pred_proba) > 0:
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['#EF553B', '#FFA15A', '#00CC96']
            labels = ['Negative', 'Neutral', 'Positive']
            
            for i, (color, label) in enumerate(zip(colors, labels)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], np.array(y_pred_proba)[:, i])
                auc = roc_auc_score(y_test_bin[:, i], np.array(y_pred_proba)[:, i])
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{label} (AUC = {auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curves for Multi-Class Sentiment Analysis', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        # Classification Report
        st.markdown("---")
        st.subheader("📋 Detailed Classification Report")
        
        if models:
            model, vectorizer = models
            # Get predictions for full test set
            report_dict = classification_report(
                y_test, 
                [np.argmax(p) for p in y_pred_proba],
                target_names=['Negative', 'Neutral', 'Positive'],
                output_dict=True,
                zero_division=0
            )
            
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df.round(3)
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']))
        
        # Feature Importance
        st.markdown("---")
        st.subheader("🔍 Top Features Influencing Predictions")
        st.info("**Note:** These are the most important words/phrases for each sentiment class")
        
        if models:
            model, vectorizer = models
            feature_names = vectorizer.get_feature_names_out()
            
            # Get feature importance from ensemble
            try:
                # Try to get feature importance from Random Forest in ensemble
                importances = model.estimators_[0].feature_importances_
                top_indices = np.argsort(importances)[-20:][::-1]
                
                top_features = [feature_names[i] for i in top_indices]
                top_importance = [importances[i] for i in top_indices]
                
                fig = px.bar(x=top_importance, y=top_features, orientation='h',
                           title='Top 20 Most Important Features',
                           labels={'x': 'Importance', 'y': 'Feature'},
                           color=top_importance, color_continuous_scale='Viridis')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Feature importance visualization not available for this model type")
    
    # Page 4: Sentiment Prediction
    elif page == "🔍 Sentiment Prediction":
        st.header("🔍 Predict Sentiment for New Reviews")
        
        if models is None:
            st.error("❌ Model not available. Please check the dataset and training process.")
            return
        
        st.markdown("""
        Enter a review below to analyze its sentiment using our trained AI model.
        The model will classify the review as **Positive**, **Neutral**, or **Negative**.
        """)
        
        # Example reviews
        st.subheader("📝 Try These Examples")
        examples = {
            "Positive Example": "This chatbot is amazing! It helps me with everything and provides accurate responses. Highly recommended!",
            "Neutral Example": "It's okay, nothing special. Works fine for basic questions but sometimes gives generic answers.",
            "Negative Example": "Terrible experience. The responses are often inaccurate and unhelpful. Very disappointed with the service.",
            "Custom Input": ""
        }
        
        selected_example = st.selectbox("Select an example or write your own:", list(examples.keys()))
        
        # Text input
        if selected_example == "Custom Input":
            user_input = st.text_area(
                "Enter Review Text:",
                height=150,
                placeholder="Example: This AI assistant is incredibly helpful and saves me so much time!"
            )
        else:
            user_input = st.text_area(
                "Enter Review Text:",
                value=examples[selected_example],
                height=150
            )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        if predict_button and user_input.strip():
            with st.spinner("🤖 Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(models, user_input)
            
            st.markdown("---")
            st.success("✅ Analysis Complete!")
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎯 Predicted Sentiment")
                
                # Color-coded sentiment display
                color_map = {
                    'Positive': ('green', '🟢'),
                    'Neutral': ('orange', '🟡'),
                    'Negative': ('red', '🔴')
                }
                
                color, emoji = color_map.get(sentiment, ('blue', '⚪'))
                
                st.markdown(f"""
                    <div style='text-align: center; padding: 30px; 
                         background-color: {color}22; border-radius: 10px; 
                         border: 3px solid {color};'>
                        <h1 style='color: {color}; margin: 0;'>{emoji} {sentiment}</h1>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("📊 Confidence Scores")
                
                if confidence:
                    # Sort by confidence
                    sorted_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
                    
                    for sent, score in sorted_conf:
                        color = color_map.get(sent, ('blue', '⚪'))[0]
                        st.markdown(f"**{sent}**")
                        st.progress(score, text=f"{score:.1%}")
                        st.markdown("")
            
            # Show processed text
            st.markdown("---")
            with st.expander("🔍 View Preprocessed Text"):
                preprocessor = TextPreprocessor()
                processed = preprocessor.preprocess(user_input)
                st.code(processed, language="text")
                st.caption("This is how the model sees your review after preprocessing")
        
        elif predict_button:
            st.warning("⚠️ Please enter some text to analyze!")
        
        # Batch Prediction
        st.markdown("---")
        st.subheader("📦 Batch Prediction")
        st.info("Upload a CSV file with a 'review' column to predict sentiment for multiple reviews")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                if 'review' in batch_df.columns:
                    st.success(f"✅ Loaded {len(batch_df)} reviews")
                    
                    if st.button("🚀 Predict All", type="primary"):
                        with st.spinner("Analyzing all reviews..."):
                            predictions = []
                            confidences = []
                            
                            for review in batch_df['review']:
                                sent, conf = predict_sentiment(models, str(review))
                                predictions.append(sent)
                                confidences.append(conf.get(sent, 0))
                            
                            batch_df['predicted_sentiment'] = predictions
                            batch_df['confidence'] = confidences
                        
                        st.success("✅ Predictions Complete!")
                        st.dataframe(batch_df)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="sentiment_predictions.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ CSV must contain a 'review' column")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    # Page 5: Key Insights
    else:  # Key Insights page
        st.header("💡 Key Insights & Recommendations")
        
        # Key Questions Analysis
        st.subheader("🔑 Answers to Key Business Questions")
        
        # Question 1: Overall sentiment
        st.markdown("### 1️⃣ What is the overall sentiment of user reviews?")
        sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive Reviews", f"{sentiment_pct.get('Positive', 0):.1f}%", 
                   delta="Good" if sentiment_pct.get('Positive', 0) > 50 else "Needs Improvement")
        col2.metric("Neutral Reviews", f"{sentiment_pct.get('Neutral', 0):.1f}%")
        col3.metric("Negative Reviews", f"{sentiment_pct.get('Negative', 0):.1f}%",
                   delta="Concerning" if sentiment_pct.get('Negative', 0) > 30 else "Acceptable")
        
        # Question 2: Sentiment vs Rating
        st.markdown("### 2️⃣ How does sentiment vary by rating?")
        rating_sentiment = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
        
        fig = px.bar(rating_sentiment.reset_index().melt(id_vars='rating'),
                    x='rating', y='value', color='sentiment',
                    title='Sentiment Distribution by Rating',
                    labels={'value': 'Percentage (%)', 'rating': 'Star Rating'},
                    color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Insight:** Check for mismatches between ratings and sentiment. 
        For example, do some 5-star reviews express negative sentiment?
        """)
        
        # Question 3: Keywords by sentiment
        st.markdown("### 3️⃣ Which keywords are most associated with each sentiment?")
        
        col1, col2, col3 = st.columns(3)
        
        for col, sentiment, color in zip([col1, col2, col3], 
                                        ['Positive', 'Neutral', 'Negative'],
                                        ['green', 'orange', 'red']):
            with col:
                st.markdown(f"**{sentiment} Keywords**")
                text = ' '.join(df[df['sentiment'] == sentiment]['processed_review'].dropna())
                words = Counter(text.split()).most_common(10)
                
                if words:
                    word_df = pd.DataFrame(words, columns=['Word', 'Frequency'])
                    fig = px.bar(word_df, x='Frequency', y='Word', orientation='h',
                               color_discrete_sequence=[color])
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Question 4: Sentiment over time
        st.markdown("### 4️⃣ How has sentiment changed over time?")
        
        if 'date' in df.columns and df['date'].notna().any():
            df_temp = df.dropna(subset=['date']).copy()
            df_temp['month'] = df_temp['date'].dt.to_period('M').astype(str)
            
            sentiment_time = df_temp.groupby(['month', 'sentiment']).size().reset_index(name='count')
            
            fig = px.line(sentiment_time, x='month', y='count', color='sentiment',
                         title='Sentiment Trends Over Time',
                         markers=True,
                         color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("**Insight:** Identify peaks in dissatisfaction or satisfaction")
        
        # Question 5: Verified users sentiment
        st.markdown("### 5️⃣ Do verified users leave more positive reviews?")
        
        if 'verified_purchase' in df.columns:
            verified_sentiment = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index') * 100
            
            fig = px.bar(verified_sentiment.reset_index().melt(id_vars='verified_purchase'),
                        x='verified_purchase', y='value', color='sentiment',
                        title='Sentiment by Verification Status',
                        labels={'value': 'Percentage (%)', 'verified_purchase': 'Verified Purchase'},
                        color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'},
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("**Insight:** Compare satisfaction between paying and free users")
        
        # Question 6: Review length vs sentiment
        st.markdown("### 6️⃣ Are longer reviews more negative?")
        
        length_sentiment = df.groupby('sentiment')['review_length'].mean().reset_index()
        
        fig = px.bar(length_sentiment, x='sentiment', y='review_length',
                    title='Average Review Length by Sentiment',
                    color='sentiment',
                    color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("**Insight:** Negative reviews tend to be longer as users explain their frustrations")
        
        # Question 7: Location sentiment
        st.markdown("### 7️⃣ Which locations show the most positive sentiment?")
        
        if 'location' in df.columns:
            location_sentiment = df.groupby('location')['sentiment'].apply(
                lambda x: (x == 'Positive').sum() / len(x) * 100
            ).sort_values(ascending=False).head(10).reset_index()
            location_sentiment.columns = ['location', 'positive_pct']
            
            fig = px.bar(location_sentiment, x='location', y='positive_pct',
                        title='Top 10 Locations by Positive Sentiment %',
                        color='positive_pct', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        # Question 8: Platform sentiment
        st.markdown("### 8️⃣ Platform sentiment comparison")
        
        if 'platform' in df.columns:
            platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
            
            fig = px.bar(platform_sentiment.reset_index().melt(id_vars='platform'),
                        x='platform', y='value', color='sentiment',
                        title='Sentiment Distribution by Platform',
                        color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'},
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Question 9: Version sentiment
        st.markdown("### 9️⃣ Which versions have better sentiment?")
        
        if 'version' in df.columns:
            version_sentiment = df.groupby('version')['sentiment'].apply(
                lambda x: (x == 'Positive').sum() / len(x) * 100
            ).sort_values(ascending=False).reset_index()
            version_sentiment.columns = ['version', 'positive_pct']
            
            fig = px.bar(version_sentiment, x='version', y='positive_pct',
                        title='Positive Sentiment % by ChatGPT Version',
                        color='positive_pct', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        # Question 10: Negative feedback themes
        st.markdown("### 🔟 What are the most common negative feedback themes?")
        
        negative_reviews = df[df['sentiment'] == 'Negative']['processed_review'].dropna()
        if len(negative_reviews) > 0:
            all_words = ' '.join(negative_reviews).split()
            common_complaints = Counter(all_words).most_common(20)
            
            complaint_df = pd.DataFrame(common_complaints, columns=['Theme', 'Frequency'])
            
            fig = px.treemap(complaint_df, path=['Theme'], values='Frequency',
                           title='Common Themes in Negative Reviews',
                           color='Frequency', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("🎯 Actionable Recommendations")
        
        recommendations = []
        
        # Calculate key metrics
        neg_pct = sentiment_pct.get('Negative', 0)
        pos_pct = sentiment_pct.get('Positive', 0)
        
        if neg_pct > 30:
            recommendations.append({
                'priority': '🔴 High',
                'area': 'Customer Satisfaction',
                'issue': f'{neg_pct:.1f}% negative reviews detected',
                'action': 'Conduct deep-dive analysis of negative feedback and prioritize top complaints'
            })
        
        if 'platform' in df.columns:
            platform_neg = df.groupby('platform')['sentiment'].apply(lambda x: (x == 'Negative').sum() / len(x) * 100)
            if platform_neg.max() - platform_neg.min() > 10:
                worse_platform = platform_neg.idxmax()
                recommendations.append({
                    'priority': '🟡 Medium',
                    'area': 'Platform Experience',
                    'issue': f'{worse_platform} platform has more negative reviews',
                    'action': f'Investigate {worse_platform}-specific issues and improve UX'
                })
        
        if pos_pct > 60:
            recommendations.append({
                'priority': '🟢 Opportunity',
                'area': 'Marketing',
                'issue': f'{pos_pct:.1f}% positive sentiment - strong user satisfaction',
                'action': 'Leverage positive reviews for marketing campaigns and testimonials'
            })
        
        if 'version' in df.columns:
            version_sent = df.groupby('version')['sentiment'].apply(lambda x: (x == 'Positive').sum() / len(x) * 100)
            if len(version_sent) > 1:
                best_version = version_sent.idxmax()
                worst_version = version_sent.idxmin()
                if version_sent[best_version] - version_sent[worst_version] > 15:
                    recommendations.append({
                        'priority': '🟡 Medium',
                        'area': 'Product Development',
                        'issue': f'Version {worst_version} performing worse than {best_version}',
                        'action': 'Analyze features that made version {best_version} successful'
                    })
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"Recommendation {i}: {rec['area']}", expanded=True):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**Priority:** {rec['priority']}")
                        st.markdown(f"**Area:** {rec['area']}")
                    with col2:
                        st.markdown(f"**Issue:** {rec['issue']}")
                        st.markdown(f"**Action:** {rec['action']}")
        
        # Summary
        st.markdown("---")
        st.success("""
        ### 📌 Summary
        
        ✅ **Model successfully trained** with {:.1f}% accuracy
        
        ✅ **{} reviews analyzed** across multiple dimensions
        
        ✅ **Key insights identified** for business decision-making
        
        ✅ **Actionable recommendations** provided for improvement
        
        **Next Steps:**
        1. Implement high-priority recommendations
        2. Monitor sentiment trends regularly
        3. Set up automated alerts for sentiment drops
        4. Continue collecting feedback for model improvement
        """.format(accuracy * 100, len(df)))

def main():
    """Main application entry point."""
    
    # Initialize NLTK
    initialize_nltk()
    
    # Load data
    file_path = 'chatgpt_style_reviews_dataset.xlsx'
    df = load_data(file_path)
    
    if df is None:
        st.error("❌ Failed to load dataset. Please check the file path.")
        return
    
    # Preprocess data
    with st.spinner("🔄 Preprocessing data..."):
        df = preprocess_data(df)
    
    if len(df) == 0:
        st.error("❌ No valid data after preprocessing.")
        return
    
    # Train model
    with st.spinner("🤖 Training sentiment analysis model..."):
        result = train_ensemble_model(df)
        
        if result and len(result) == 9:
            models, accuracy, precision, recall, f1, roc_auc, cm, y_test, y_pred_proba = result
        else:
            st.error("❌ Model training failed.")
            return
    
    # Generate visualizations
    with st.spinner("🎨 Generating visualizations..."):
        pos_wc, neg_wc, neu_wc = generate_word_clouds(df)
    
    # Create dashboard
    create_streamlit_dashboard(df, models, accuracy, precision, recall, f1, 
                               roc_auc, cm, y_test, y_pred_proba,
                               pos_wc, neg_wc, neu_wc)

if __name__ == "__main__":
    main()