#!/usr/bin/env python3
"""
🛡️ CyberShield AML Detection System
A comprehensive, enterprise-ready anti-money laundering detection platform
Built for CIIS 2025 National CyberShield Hackathon
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import shap
import joblib
import sqlite3
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from faker import Faker
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION & SETUP ====================

@dataclass
class Config:
    """System configuration with environment variable support"""
    DATA_PATH: str = "transactions.csv"
    MODEL_PATH: str = "aml_model.pkl"
    QUEUE_PATH: str = "investigation_queue.db"
    HIGH_RISK_THRESHOLD: float = 8.0
    MEDIUM_RISK_THRESHOLD: float = 6.0
    MAX_NETWORK_NODES: int = 100
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "cybershield-demo-key"  # In production: use .env

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    config = Config()
    for field in config.__dataclass_fields__:
        env_val = os.getenv(field)
        if env_val:
            setattr(config, field, type(getattr(config, field))(env_val))
except ImportError:
    config = Config()

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ==================== DATA STORAGE LAYER ====================

class StorageInterface:
    """Abstract storage interface for pluggable backends"""
    
    def save_queue_item(self, item: Dict) -> bool:
        raise NotImplementedError
    
    def load_queue_items(self) -> List[Dict]:
        raise NotImplementedError
    
    def update_queue_item(self, item_id: str, updates: Dict) -> bool:
        raise NotImplementedError

class SQLiteStorage(StorageInterface):
    """SQLite implementation with connection pooling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS investigation_queue (
                        id TEXT PRIMARY KEY,
                        alert_id TEXT UNIQUE NOT NULL,
                        priority TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'NEW',
                        assigned_to TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        notes TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_queue_item(self, item: Dict) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO investigation_queue 
                    (id, alert_id, priority, status, assigned_to, notes, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    item.get('id', item['alert_id']),
                    item['alert_id'],
                    item['priority'],
                    item.get('status', 'NEW'),
                    item.get('assigned_to', ''),
                    item.get('notes', ''),
                    json.dumps(item.get('metadata', {}))
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save queue item: {e}")
            return False
    
    def load_queue_items(self) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM investigation_queue 
                    ORDER BY created_at DESC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to load queue items: {e}")
            return []
    
    def update_queue_item(self, item_id: str, updates: Dict) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values()) + [item_id]
                conn.execute(f"""
                    UPDATE investigation_queue 
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
                    WHERE alert_id = ?
                """, values)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update queue item: {e}")
            return False

# Initialize storage
storage = SQLiteStorage(config.QUEUE_PATH)

# ==================== DATA GENERATION & PREPROCESSING ====================

class DataGenerator:
    """Generate synthetic transaction data for demonstration"""
    
    def __init__(self, random_state: int = 42):
        self.fake = Faker()
        self.fake.seed_instance(random_state)
        np.random.seed(random_state)
    
    def generate_transactions(self, n: int = 5000) -> pd.DataFrame:
        """Generate synthetic transaction dataset"""
        logger.info(f"Generating {n} synthetic transactions...")
        
        transactions = []
        for i in range(n):
            # Base transaction
            amount = np.random.lognormal(mean=6, sigma=2)
            hour = np.random.randint(0, 24)
            is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])
            is_international = np.random.choice([0, 1], p=[0.8, 0.2])
            
            # Suspicious pattern injection (10% of transactions)
            is_suspicious = 0
            if np.random.random() < 0.1:
                is_suspicious = 1
                # Make it more suspicious
                if np.random.random() < 0.5:
                    amount = np.random.choice([9500, 9800, 9900, 9950, 9999])  # Threshold avoidance
                if np.random.random() < 0.3:
                    hour = np.random.choice([22, 23, 0, 1, 2, 3])  # Night transactions
                if np.random.random() < 0.4:
                    amount = np.random.choice([50000, 75000, 100000])  # Large amounts
                if np.random.random() < 0.6:
                    is_international = 1  # Cross-border
            
            transaction = {
                'id': f'TXN_{i:06d}',
                'amount': round(amount, 2),
                'sender': f'ACC_{np.random.randint(1000, 9999)}',
                'receiver': f'ACC_{np.random.randint(1000, 9999)}',
                'hour': hour,
                'is_weekend': is_weekend,
                'is_international': is_international,
                'is_suspicious': is_suspicious,
                'timestamp': self.fake.date_time_between(start_date='-30d', end_date='now')
            }
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        logger.info(f"Generated {len(df)} transactions ({df['is_suspicious'].sum()} suspicious)")
        return df

class FeatureEngineer:
    """Advanced feature engineering for AML detection"""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        logger.info("Engineering features...")
        
        df = df.copy()
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
        df['is_large_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Time-based features
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Account behavior features
        sender_stats = df.groupby('sender').agg({
            'amount': ['count', 'sum', 'mean'],
            'is_international': 'mean'
        }).round(2)
        sender_stats.columns = ['sender_tx_count', 'sender_total_amount', 'sender_avg_amount', 'sender_intl_ratio']
        
        receiver_stats = df.groupby('receiver').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        receiver_stats.columns = ['receiver_tx_count', 'receiver_total_amount', 'receiver_avg_amount']
        
        # Merge account statistics
        df = df.merge(sender_stats, left_on='sender', right_index=True, how='left')
        df = df.merge(receiver_stats, left_on='receiver', right_index=True, how='left')
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

# ==================== DETECTION ENGINES ====================

class RuleEngine:
    """Enhanced rule-based detection system"""
    
    def __init__(self):
        self.rules = {
            'threshold_avoidance': {'weight': 4.0, 'threshold': 9500},
            'large_amount': {'weight': 3.0, 'threshold': 50000},
            'night_transaction': {'weight': 2.0},
            'weekend_international': {'weight': 2.5},
            'round_amount_large': {'weight': 2.0, 'threshold': 10000},
            'high_frequency_sender': {'weight': 1.5, 'threshold': 10}
        }
    
    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        """Evaluate rule-based risk scores"""
        logger.info("Evaluating rule-based detection...")
        
        scores = np.zeros(len(df))
        
        # Rule 1: Threshold avoidance (amounts just below reporting thresholds)
        threshold_avoid = (df['amount'] >= self.rules['threshold_avoidance']['threshold']) & (df['amount'] < 10000)
        scores += threshold_avoid * self.rules['threshold_avoidance']['weight']
        
        # Rule 2: Large amounts
        large_amounts = df['amount'] >= self.rules['large_amount']['threshold']
        scores += large_amounts * self.rules['large_amount']['weight']
        
        # Rule 3: Night transactions
        night_tx = df['is_night'] == 1
        scores += night_tx * self.rules['night_transaction']['weight']
        
        # Rule 4: Weekend + international
        weekend_intl = (df['is_weekend'] == 1) & (df['is_international'] == 1)
        scores += weekend_intl * self.rules['weekend_international']['weight']
        
        # Rule 5: Large round amounts
        large_round = (df['is_round_amount'] == 1) & (df['amount'] >= self.rules['round_amount_large']['threshold'])
        scores += large_round * self.rules['round_amount_large']['weight']
        
        # Rule 6: High-frequency senders
        high_freq = df['sender_tx_count'] >= self.rules['high_frequency_sender']['threshold']
        scores += high_freq * self.rules['high_frequency_sender']['weight']
        
        logger.info(f"Rule-based scores computed. Range: {scores.min():.2f} - {scores.max():.2f}")
        return scores

class MLEngine:
    """Machine learning detection engine with ensemble models"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        feature_cols = [
            'amount_log', 'hour', 'is_weekend', 'is_international',
            'is_round_amount', 'is_large_amount', 'is_night', 'is_business_hours',
            'sender_tx_count', 'sender_avg_amount', 'sender_intl_ratio',
            'receiver_tx_count', 'receiver_avg_amount', 'amount_zscore'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].fillna(0)
        return X.values
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train ensemble of ML models"""
        logger.info("Training ML ensemble...")
        
        X = self.prepare_features(df)
        y = df['is_suspicious'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        
        # Train Isolation Forest (unsupervised anomaly detection)
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1, random_state=42, n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_train_scaled)
        
        # Evaluate models
        rf_score = self.models['random_forest'].score(X_test_scaled, y_test)
        
        # Anomaly detection evaluation (approximate)
        iso_predictions = self.models['isolation_forest'].predict(X_test_scaled)
        iso_score = np.mean((iso_predictions == -1) == y_test)  # Rough approximation
        
        self.is_trained = True
        
        metrics = {
            'random_forest_accuracy': rf_score,
            'isolation_forest_accuracy': iso_score,
            'total_samples': len(df),
            'positive_samples': y.sum()
        }
        
        logger.info(f"ML training complete. RF Accuracy: {rf_score:.3f}")
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ML-based risk scores"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Random Forest probabilities
        rf_proba = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
        
        # Isolation Forest anomaly scores (convert to 0-1 scale)
        iso_scores = self.models['isolation_forest'].decision_function(X_scaled)
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        iso_scores_norm = 1 - iso_scores_norm  # Invert so higher = more anomalous
        
        return rf_proba, iso_scores_norm

class RiskScorer:
    """Ensemble risk scoring system"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'rules': 0.3,
            'random_forest': 0.4,
            'isolation_forest': 0.3
        }
    
    def compute_risk_score(self, rule_scores: np.ndarray, rf_scores: np.ndarray, 
                          iso_scores: np.ndarray) -> np.ndarray:
        """Compute ensemble risk scores"""
        # Normalize all scores to 0-10 scale
        rule_scores_norm = (rule_scores / rule_scores.max()) * 10 if rule_scores.max() > 0 else rule_scores
        rf_scores_norm = rf_scores * 10
        iso_scores_norm = iso_scores * 10
        
        # Weighted ensemble
        final_scores = (
            rule_scores_norm * self.weights['rules'] +
            rf_scores_norm * self.weights['random_forest'] +
            iso_scores_norm * self.weights['isolation_forest']
        )
        
        return np.clip(final_scores, 0, 10)
    
    def categorize_risk(self, scores: np.ndarray, 
                       high_threshold: float = 8.0, 
                       medium_threshold: float = 6.0) -> np.ndarray:
        """Categorize risk scores into HIGH/MEDIUM/LOW"""
        categories = np.where(
            scores >= high_threshold, 'HIGH',
            np.where(scores >= medium_threshold, 'MEDIUM', 'LOW')
        )
        return categories

# ==================== EXPLAINABLE AI ====================

class XAIExplainer:
    """Explainable AI system with multiple explanation methods"""
    
    def __init__(self):
        self.explainers = {}
    
    def initialize(self, models: Dict, feature_names: List[str]):
        """Initialize explainers for trained models"""
        try:
            if 'random_forest' in models:
                self.explainers['shap'] = shap.TreeExplainer(models['random_forest'])
            self.feature_names = feature_names
            logger.info("XAI explainers initialized")
        except Exception as e:
            logger.error(f"Failed to initialize XAI: {e}")
    
    def explain_prediction(self, X: np.ndarray, transaction_id: str) -> Dict:
        """Generate explanation for a single prediction"""
        try:
            if 'shap' not in self.explainers:
                return {'error': 'SHAP explainer not available'}
            
            # Get SHAP values
            shap_values = self.explainers['shap'].shap_values(X)
            
            # For binary classification, use positive class
            if len(shap_values) == 2:
                explanation_values = shap_values[1][0]
            else:
                explanation_values = shap_values[0]
            
            # Create feature importance ranking
            feature_importance = [
                {
                    'feature': self.feature_names[i],
                    'importance': float(explanation_values[i]),
                    'direction': 'increases' if explanation_values[i] > 0 else 'decreases'
                }
                for i in range(len(explanation_values))
            ]
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
            
            return {
                'transaction_id': transaction_id,
                'feature_importance': feature_importance[:10],  # Top 10 features
                'total_impact': float(np.sum(explanation_values))
            }
        
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return {'error': str(e)}

# ==================== NETWORK ANALYSIS ====================

class NetworkAnalyzer:
    """Advanced network analysis for suspicious transaction patterns"""
    
    def __init__(self):
        self.graph = None
        self.suspicious_patterns = {}
    
    def build_network(self, df: pd.DataFrame, min_risk_score: float = 6.0) -> nx.DiGraph:
        """Build transaction network graph"""
        logger.info("Building transaction network...")
        
        # Filter high-risk transactions
        high_risk_df = df[df['final_risk_score'] >= min_risk_score].copy()
        
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Add edges with attributes
        for _, row in high_risk_df.iterrows():
            self.graph.add_edge(
                row['sender'], 
                row['receiver'],
                weight=row['amount'],
                risk_score=row['final_risk_score'],
                transaction_id=row['id']
            )
        
        logger.info(f"Network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def detect_patterns(self) -> Dict:
        """Detect suspicious patterns in the network"""
        if self.graph is None:
            return {}
        
        patterns = {}
        
        try:
            # Circular flows (cycles)
            cycles = list(nx.simple_cycles(self.graph))
            patterns['circular_flows'] = cycles[:5]  # Top 5 cycles
            
            # Hub accounts (high degree centrality)
            centrality = nx.degree_centrality(self.graph)
            hub_threshold = np.percentile(list(centrality.values()), 90) if centrality else 0
            patterns['hub_accounts'] = [
                {'account': node, 'centrality': score}
                for node, score in centrality.items()
                if score >= hub_threshold
            ][:10]
            
            # Connected components
            components = list(nx.weakly_connected_components(self.graph))
            patterns['isolated_clusters'] = [
                list(component) for component in components
                if 3 <= len(component) <= 10
            ][:5]
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            patterns['error'] = str(e)
        
        self.suspicious_patterns = patterns
        return patterns
    
    def create_visualization(self) -> go.Figure:
        """Create interactive network visualization"""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return go.Figure().add_annotation(text="No network data available", showarrow=False)
        
        try:
            # Limit nodes for performance
            if self.graph.number_of_nodes() > config.MAX_NETWORK_NODES:
                # Keep top nodes by degree
                top_nodes = sorted(self.graph.nodes(), 
                                 key=lambda x: self.graph.degree(x), 
                                 reverse=True)[:config.MAX_NETWORK_NODES]
                subgraph = self.graph.subgraph(top_nodes)
            else:
                subgraph = self.graph
            
            # Calculate layout
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
            
            # Create traces
            edge_x, edge_y = [], []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = [pos[node][0] for node in subgraph.nodes()]
            node_y = [pos[node][1] for node in subgraph.nodes()]
            node_text = [f"{node}<br>Degree: {subgraph.degree(node)}" for node in subgraph.nodes()]
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_text,
                text=[node.split('_')[1] if '_' in node else node for node in subgraph.nodes()],
                textposition="middle center",
                marker=dict(
                    size=10,
                    color='lightblue',
                    line=dict(width=2, color='black')
                ),
                showlegend=False
            ))
            
            fig.update_layout(
                title="Transaction Network Graph",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Nodes represent accounts, edges represent transactions",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Network visualization failed: {e}")
            return go.Figure().add_annotation(text=f"Visualization error: {e}", showarrow=False)

# ==================== STREAMLIT UI ====================

def configure_page():
    """Configure Streamlit page settings and styling"""
    st.set_page_config(
        page_title="🛡️ CyberShield AML System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .status-new { background-color: #ffeaa7; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .status-progress { background-color: #74b9ff; padding: 0.2rem 0.5rem; border-radius: 4px; color: white; }
    .status-resolved { background-color: #00b894; padding: 0.2rem 0.5rem; border-radius: 4px; color: white; }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    """Display main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ CyberShield AML Detection System</h1>
        <p>Military-grade Anti-Money Laundering Detection Platform | CIIS 2025 CyberShield Hackathon</p>
    </div>
    """, unsafe_allow_html=True)

def show_system_metrics(df: pd.DataFrame):
    """Display key system metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_transactions = len(df)
    high_risk = len(df[df['risk_category'] == 'HIGH'])
    medium_risk = len(df[df['risk_category'] == 'MEDIUM'])
    total_volume = df['amount'].sum()
    avg_risk = df['final_risk_score'].mean()
    
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("🔴 High Risk", f"{high_risk:,}", f"{high_risk/total_transactions*100:.1f}%")
    col3.metric("🟡 Medium Risk", f"{medium_risk:,}", f"{medium_risk/total_transactions*100:.1f}%")
    col4.metric("💰 Total Volume", f"${total_volume:,.0f}")
    col5.metric("📊 Avg Risk Score", f"{avg_risk:.2f}/10")

def show_overview_tab(df: pd.DataFrame):
    """Overview dashboard tab"""
    st.subheader("📊 System Overview")
    
    # Recent alerts
    st.subheader("🚨 Recent High-Risk Alerts")
    high_risk_alerts = df[df['risk_category'] == 'HIGH'].nlargest(10, 'final_risk_score')
    
    for _, alert in high_risk_alerts.iterrows():
        with st.expander(f"🔴 {alert['id']} - Risk Score: {alert['final_risk_score']:.2f}"):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Amount:** ${alert['amount']:,.2f}")
            col2.write(f"**Sender:** {alert['sender']}")
            col3.write(f"**Receiver:** {alert['receiver']}")
            
            if st.button(f"Add to Investigation Queue", key=f"queue_{alert['id']}"):
                storage.save_queue_item({
                    'alert_id': alert['id'],
                    'priority': 'HIGH',
                    'status': 'NEW',
                    'metadata': {
                        'amount': alert['amount'],
                        'risk_score': alert['final_risk_score']
                    }
                })
                st.success(f"Added {alert['id']} to investigation queue")

def show_queue_tab():
    """Investigation queue management tab"""
    st.subheader("🗂️ Cyber-Cell Investigation Queue")
    
    # Queue controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Queue"):
            st.rerun()
    
    with col2:
        if st.button("📥 Sync High-Risk Alerts"):
            # This would sync new high-risk alerts to queue
            st.info("Alert sync functionality would be implemented here")
    
    with col3:
        if st.button("📊 Export Queue"):
            queue_items = storage.load_queue_items()
            if queue_items:
                queue_df = pd.DataFrame(queue_items)
                csv = queue_df.to_csv(index=False)
                st.download_button(
                    "Download Queue CSV",
                    csv,
                    f"investigation_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
    
    # Display queue items
    queue_items = storage.load_queue_items()
    
    if not queue_items:
        st.info("No items in investigation queue. Add alerts from the Overview tab.")
        return
    
    # Queue statistics
    queue_df = pd.DataFrame(queue_items)
    status_counts = queue_df['status'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items", len(queue_items))
    col2.metric("New", status_counts.get('NEW', 0))
    col3.metric("In Progress", status_counts.get('IN_PROGRESS', 0))
    col4.metric("Resolved", status_counts.get('RESOLVED', 0))
    
    # Queue items
    st.subheader("Queue Items")
    
    for item in queue_items:
        status_class = f"status-{item['status'].lower().replace('_', '')}"
        
        with st.expander(f"🎯 {item['alert_id']} - {item['priority']} Priority"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Status:** {item['status']}")
                st.write(f"**Created:** {item['created_at']}")
                st.write(f"**Updated:** {item['updated_at']}")
                if item.get('assigned_to'):
                    st.write(f"**Assigned to:** {item['assigned_to']}")
                if item.get('notes'):
                    st.write(f"**Notes:** {item['notes']}")
            
            with col2:
                # Update controls
                new_status = st.selectbox(
                    "Update Status",
                    ["NEW", "IN_PROGRESS", "RESOLVED", "ESCALATED"],
                    index=["NEW", "IN_PROGRESS", "RESOLVED", "ESCALATED"].index(item['status']),
                    key=f"status_{item['alert_id']}"
                )
                
                new_assignee = st.text_input(
                    "Assign to",
                    value=item.get('assigned_to', ''),
                    key=f"assignee_{item['alert_id']}"
                )
                
                new_notes = st.text_area(
                    "Investigation Notes",
                    value=item.get('notes', ''),
                    key=f"notes_{item['alert_id']}"
                )
                
                if st.button("Update", key=f"update_{item['alert_id']}"):
                    updates = {
                        'status': new_status,
                        'assigned_to': new_assignee,
                        'notes': new_notes
                    }
                    
                    if storage.update_queue_item(item['alert_id'], updates):
                        st.success("Item updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update item")

def show_xai_tab(df: pd.DataFrame, ml_engine: MLEngine):
    """Explainable AI tab"""
    st.subheader("🔍 Explainable AI Analysis")
    
    # Transaction selector
    high_risk_transactions = df[df['risk_category'] != 'LOW']['id'].tolist()
    
    if not high_risk_transactions:
        st.info("No medium or high-risk transactions available for analysis")
        return
    
    selected_txn_id = st.selectbox(
        "Select Transaction for Analysis",
        high_risk_transactions,
        help="Choose a transaction to see detailed AI explanations"
    )
    
    # Get transaction details
    transaction = df[df['id'] == selected_txn_id].iloc[0]
    
    # Display transaction info
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{transaction['final_risk_score']:.2f}/10")
    col2.metric("Risk Category", transaction['risk_category'])
    col3.metric("Amount", f"${transaction['amount']:,.2f}")
    
    st.write(f"**Transaction Details:**")
    st.write(f"Sender: {transaction['sender']} → Receiver: {transaction['receiver']}")
    st.write(f"Time: Hour {transaction['hour']:02d}:00, Weekend: {'Yes' if transaction['is_weekend'] else 'No'}")
    st.write(f"International: {'Yes' if transaction['is_international'] else 'No'}")
    
    # Generate explanation if models are available
    if hasattr(ml_engine, 'models') and ml_engine.is_trained:
        try:
            # Prepare data for explanation
            X = ml_engine.prepare_features(df[df['id'] == selected_txn_id])
            
            # Initialize XAI explainer if not already done
            if not hasattr(st.session_state, 'xai_explainer'):
                st.session_state.xai_explainer = XAIExplainer()
                st.session_state.xai_explainer.initialize(ml_engine.models, ml_engine.feature_names)
            
            # Generate explanation
            explanation = st.session_state.xai_explainer.explain_prediction(X, selected_txn_id)
            
            if 'error' not in explanation:
                st.subheader("🎯 Feature Importance Analysis")
                
                # Display top contributing features
                for i, feature in enumerate(explanation['feature_importance'][:5]):
                    direction_color = "🔴" if feature['importance'] > 0 else "🔵"
                    st.write(f"{direction_color} **{feature['feature']}**: {feature['direction']} risk by {abs(feature['importance']):.3f}")
                
                # Create visualization
                feature_data = explanation['feature_importance'][:8]
                features = [f['feature'] for f in feature_data]
                importances = [f['importance'] for f in feature_data]
                
                fig = go.Figure(data=go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    marker_color=['red' if x > 0 else 'blue' for x in importances]
                ))
                
                fig.update_layout(
                    title="Feature Impact on Risk Score",
                    xaxis_title="SHAP Value (Impact)",
                    yaxis_title="Feature",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment summary
                st.subheader("📋 Risk Assessment Summary")
                
                risk_factors = []
                if transaction['is_large_amount']:
                    risk_factors.append("Large transaction amount")
                if transaction['is_round_amount']:
                    risk_factors.append("Round number amount")
                if transaction['is_night']:
                    risk_factors.append("Night-time transaction")
                if transaction['is_international']:
                    risk_factors.append("International transfer")
                
                if risk_factors:
                    st.write("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                
                # Recommendations
                if transaction['final_risk_score'] >= config.HIGH_RISK_THRESHOLD:
                    st.error("🚨 **HIGH PRIORITY**: Immediate investigation required")
                elif transaction['final_risk_score'] >= config.MEDIUM_RISK_THRESHOLD:
                    st.warning("⚠️ **MEDIUM PRIORITY**: Investigation recommended")
                else:
                    st.info("ℹ️ **LOW PRIORITY**: Monitor only")
            
            else:
                st.error(f"Explanation failed: {explanation['error']}")
        
        except Exception as e:
            st.error(f"Failed to generate explanation: {e}")
            logger.error(f"XAI explanation error: {e}")
    
    else:
        st.warning("⚠️ XAI explanations require trained ML models. Please run the detection pipeline first.")

def show_network_tab(df: pd.DataFrame):
    """Network analysis tab"""
    st.subheader("🔗 Network Analysis")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_risk_score = st.slider(
            "Minimum Risk Score", 
            min_value=0.0, 
            max_value=10.0, 
            value=6.0, 
            step=0.1,
            help="Filter transactions by minimum risk score"
        )
    
    with col2:
        if st.button("🔄 Build Network"):
            with st.spinner("Building network..."):
                try:
                    # Initialize network analyzer
                    if 'network_analyzer' not in st.session_state:
                        st.session_state.network_analyzer = NetworkAnalyzer()
                    
                    # Build network
                    graph = st.session_state.network_analyzer.build_network(df, min_risk_score)
                    
                    if graph.number_of_nodes() > 0:
                        st.success(f"Network built: {graph.number_of_nodes()} accounts, {graph.number_of_edges()} transactions")
                    else:
                        st.warning("No network data found with current filters")
                
                except Exception as e:
                    st.error(f"Network building failed: {e}")
                    logger.error(f"Network building error: {e}")
    
    with col3:
        if st.button("🔍 Detect Patterns"):
            if hasattr(st.session_state, 'network_analyzer') and st.session_state.network_analyzer.graph:
                with st.spinner("Detecting suspicious patterns..."):
                    patterns = st.session_state.network_analyzer.detect_patterns()
                    st.session_state.network_patterns = patterns
                    st.success("Pattern detection complete")
            else:
                st.warning("Please build network first")
    
    # Display network visualization
    if hasattr(st.session_state, 'network_analyzer') and st.session_state.network_analyzer.graph:
        
        # Network statistics
        graph = st.session_state.network_analyzer.graph
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accounts", graph.number_of_nodes())
        col2.metric("Transactions", graph.number_of_edges())
        
        if graph.number_of_nodes() > 0:
            avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            col3.metric("Avg Connections", f"{avg_degree:.1f}")
            col4.metric("Density", f"{nx.density(graph):.3f}")
        
        # Network visualization
        st.subheader("🕸️ Transaction Network")
        
        if graph.number_of_nodes() > config.MAX_NETWORK_NODES:
            st.warning(f"⚠️ Network has {graph.number_of_nodes()} nodes. Showing top {config.MAX_NETWORK_NODES} by connectivity.")
        
        try:
            fig = st.session_state.network_analyzer.create_visualization()
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Network visualization failed: {e}")
        
        # Display detected patterns
        if hasattr(st.session_state, 'network_patterns'):
            patterns = st.session_state.network_patterns
            
            st.subheader("🔍 Detected Suspicious Patterns")
            
            # Circular flows
            if patterns.get('circular_flows'):
                with st.expander(f"🔄 Circular Money Flows ({len(patterns['circular_flows'])} found)"):
                    for i, cycle in enumerate(patterns['circular_flows']):
                        cycle_str = " → ".join(cycle) + f" → {cycle[0]}"
                        st.write(f"**Flow {i+1}:** {cycle_str}")
            
            # Hub accounts
            if patterns.get('hub_accounts'):
                with st.expander(f"🎯 Hub Accounts ({len(patterns['hub_accounts'])} found)"):
                    for hub in patterns['hub_accounts']:
                        st.write(f"**{hub['account']}:** Centrality score {hub['centrality']:.3f}")
            
            # Isolated clusters
            if patterns.get('isolated_clusters'):
                with st.expander(f"🏝️ Isolated Clusters ({len(patterns['isolated_clusters'])} found)"):
                    for i, cluster in enumerate(patterns['isolated_clusters']):
                        st.write(f"**Cluster {i+1}:** {', '.join(cluster)}")
    
    else:
        st.info("Click 'Build Network' to analyze transaction relationships")

def show_settings_tab():
    """Settings and configuration tab"""
    st.subheader("⚙️ System Settings")
    
    # Detection thresholds
    st.subheader("🎯 Detection Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_high_threshold = st.slider(
            "High Risk Threshold",
            min_value=5.0,
            max_value=10.0,
            value=config.HIGH_RISK_THRESHOLD,
            step=0.1,
            help="Transactions above this score are classified as HIGH risk"
        )
    
    with col2:
        new_medium_threshold = st.slider(
            "Medium Risk Threshold", 
            min_value=3.0,
            max_value=8.0,
            value=config.MEDIUM_RISK_THRESHOLD,
            step=0.1,
            help="Transactions above this score are classified as MEDIUM risk"
        )
    
    # Model settings
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ensemble_weights = st.expander("⚖️ Ensemble Weights")
        with ensemble_weights:
            rule_weight = st.slider("Rule Engine Weight", 0.0, 1.0, 0.3, 0.1)
            rf_weight = st.slider("Random Forest Weight", 0.0, 1.0, 0.4, 0.1)
            iso_weight = st.slider("Isolation Forest Weight", 0.0, 1.0, 0.3, 0.1)
            
            total_weight = rule_weight + rf_weight + iso_weight
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum to {total_weight:.2f}, should sum to 1.0")
    
    with col2:
        data_settings = st.expander("📊 Data Settings")
        with data_settings:
            max_network_nodes = st.number_input(
                "Max Network Nodes",
                min_value=50,
                max_value=1000,
                value=config.MAX_NETWORK_NODES,
                help="Maximum nodes to display in network graph"
            )
    
    # System information
    st.subheader("ℹ️ System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write("**Configuration:**")
        st.write(f"• Data Path: `{config.DATA_PATH}`")
        st.write(f"• Model Path: `{config.MODEL_PATH}`")
        st.write(f"• Queue Path: `{config.QUEUE_PATH}`")
        st.write(f"• Log Level: `{config.LOG_LEVEL}`")
    
    with info_col2:
        st.write("**Current Session:**")
        if 'df' in st.session_state:
            st.write(f"• Transactions Loaded: {len(st.session_state.df):,}")
        if hasattr(st.session_state, 'ml_engine') and st.session_state.ml_engine.is_trained:
            st.write("• ML Models: ✅ Trained")
        else:
            st.write("• ML Models: ❌ Not trained")
        
        queue_count = len(storage.load_queue_items())
        st.write(f"• Queue Items: {queue_count}")

def show_about_tab():
    """About and help tab"""
    st.subheader("ℹ️ About CyberShield AML System")
    
    # Child-friendly explanation
    st.subheader("👦 Simple Explanation (for everyone!)")
    st.markdown("""
    **Think of this system like a super-smart cyber police station:**
    
    🏪 **The Front Desk** - All money transactions come here first (like people walking into a police station)
    
    🕵️ **The Detectives** - Our computer "detectives" look at each transaction and ask:
    - "Is this person moving money at weird times?" (like 3 AM)
    - "Are they moving LOTS of money?" (like carrying huge bags of cash)
    - "Are they trying to hide something?" (like moving $9,999 instead of $10,000)
    
    🚨 **The Alert System** - When something looks suspicious, it goes "BEEP BEEP!" and puts a red flag on it
    
    🗂️ **The Investigation Room** - Real human investigators can look at the flagged cases and decide what to do
    
    🕸️ **The Detective Board** - Like those movies where detectives connect suspects with string! We show which accounts are connected to each other
    
    📊 **The Smart Helper** - Our AI assistant explains WHY it thinks something is suspicious (like "this person came at night with a big bag of money")
    
    **The goal:** Catch the "bad guys" who try to hide dirty money and keep the banking system safe! 🛡️
    """)
    
    # Technical explanation
    st.subheader("👨‍💻 Technical Overview")
    st.markdown("""
    **CyberShield is an enterprise-grade AML detection platform with the following architecture:**
    
    ```
    Data Ingestion → Feature Engineering → Multi-Model Detection → Risk Scoring → Investigation Queue
                                                    ↓
    Network Analysis ← XAI Explanations ← Alert Management ← Real-time Dashboard
    ```
    
    **Core Components:**
    - **Rule Engine**: Domain-specific heuristics for threshold avoidance, timing anomalies
    - **ML Ensemble**: Random Forest (supervised) + Isolation Forest (unsupervised)
    - **Risk Scorer**: Weighted ensemble with configurable thresholds
    - **XAI Module**: SHAP-based feature importance explanations
    - **Network Analyzer**: Graph-based pattern detection (cycles, hubs, clusters)
    - **Investigation Queue**: CRUD interface with status tracking and notes
    - **Storage Layer**: Pluggable backend (SQLite default, PostgreSQL ready)
    
    **Risk Scoring Formula:**
    ```
    Final Risk = (0.3 × Rule Score) + (0.4 × RF Probability) + (0.3 × Anomaly Score)
    ```
    
    **Detection Patterns:**
    - Structuring/Smurfing (amounts just below $10K threshold)
    - Unusual timing (night/weekend transactions)
    - Large round amounts
    - Circular money flows
    - High-frequency senders
    - Cross-border + weekend combinations
    """)
    
    # Limitations and future work
    st.subheader("⚠️ Current Limitations & Future Pillars")
    
    limitations = [
        ("Synthetic Dataset", "Real-world data integration with schema validation"),
        ("Basic ML Models", "Advanced ensemble with graph neural networks"),
        ("SHAP-only XAI", "Multi-modal explanations with natural language"),
        ("SQLite Storage", "Distributed database with Redis caching"),
        ("Static Network Viz", "Real-time 3D visualization with GPU acceleration"),
        ("Batch Processing", "Streaming analytics with Apache Kafka"),
        ("Streamlit UI", "React-based enterprise interface with mobile support")
    ]
    
    for limitation, future in limitations:
        col1, col2 = st.columns([1, 2])
        col1.write(f"**Current:** {limitation}")
        col2.write(f"**Future:** {future}")
    
    # System stats
    st.subheader("📊 System Statistics")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Data Points", len(df))
        col2.metric("Features", len([c for c in df.columns if c.startswith(('amount', 'is_', 'sender', 'receiver'))]))
        col3.metric("Models", "2 (RF + IF)")

# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point"""
    configure_page()
    show_header()
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.processing = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Generate Demo Data", "Upload CSV File"],
            help="Choose data source for analysis"
        )
        
        # Main processing button
        if st.button("🚀 Run Detection Pipeline", type="primary"):
            st.session_state.processing = True
            run_detection_pipeline(data_source)
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Reset System"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'initialized':
                    del st.session_state[key]
            st.success("System reset complete")
            st.rerun()
        
        if st.button("📥 Load Sample Data"):
            load_sample_data()
        
        # System status
        st.subheader("🔋 System Status")
        
        data_status = "✅ Loaded" if 'df' in st.session_state else "❌ No data"
        st.write(f"**Data:** {data_status}")
        
        model_status = "✅ Trained" if hasattr(st.session_state, 'ml_engine') and getattr(st.session_state.ml_engine, 'is_trained', False) else "❌ Not trained"
        st.write(f"**Models:** {model_status}")
        
        queue_count = len(storage.load_queue_items())
        st.write(f"**Queue:** {queue_count} items")
    
    # Main content area
    if 'df' in st.session_state:
        show_system_metrics(st.session_state.df)
        
        # Tab navigation
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🗂️ Queue", 
            "🔍 Explainable AI", 
            "🔗 Network", 
            "⚙️ Settings", 
            "ℹ️ About"
        ])
        
        with tab1:
            show_overview_tab(st.session_state.df)
        
        with tab2:
            show_queue_tab()
        
        with tab3:
            ml_engine = getattr(st.session_state, 'ml_engine', None)
            show_xai_tab(st.session_state.df, ml_engine)
        
        with tab4:
            show_network_tab(st.session_state.df)
        
        with tab5:
            show_settings_tab()
        
        with tab6:
            show_about_tab()
    
    else:
        # Welcome screen
        st.subheader("🚀 Welcome to CyberShield AML Detection System")
        st.write("Please select a data source and click 'Run Detection Pipeline' to begin.")
        
        # Quick start guide
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
            **Step 1:** Choose data source (Generate Demo Data recommended for first time)
            
            **Step 2:** Click "🚀 Run Detection Pipeline" in the sidebar
            
            **Step 3:** Wait for processing to complete (30-60 seconds)
            
            **Step 4:** Explore the tabs:
            - **Overview**: See detected alerts and add to investigation queue
            - **Queue**: Manage investigation cases with status tracking
            - **Explainable AI**: Understand why transactions were flagged
            - **Network**: Visualize suspicious account relationships
            - **Settings**: Configure detection thresholds
            - **About**: Learn how the system works
            
            **Demo Flow (5 minutes):**
            1. Generate demo data and run pipeline
            2. View high-risk alerts in Overview tab
            3. Add some alerts to investigation queue
            4. Go to Queue tab and update investigation status
            5. Select a transaction in XAI tab to see explanations
            6. Build network graph and detect suspicious patterns
            """)

def load_sample_data():
    """Load sample data for quick testing"""
    try:
        # Generate small sample dataset
        generator = DataGenerator()
        df = generator.generate_transactions(1000)
        
        # Basic feature engineering
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        
        # Add dummy risk scores for demo
        df['final_risk_score'] = np.random.beta(2, 5, len(df)) * 10
        df['risk_category'] = pd.cut(df['final_risk_score'], 
                                   bins=[0, config.MEDIUM_RISK_THRESHOLD, config.HIGH_RISK_THRESHOLD, 10],
                                   labels=['LOW', 'MEDIUM', 'HIGH'])
        
        st.session_state.df = df
        st.success("Sample data loaded successfully!")
        
    except Exception as e:
        st.error(f"Failed to load sample data: {e}")
        logger.error(f"Sample data loading error: {e}")

def run_detection_pipeline(data_source: str):
    """Run the complete detection pipeline"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load/Generate Data
        status_text.text("📥 Loading data...")
        progress_bar.progress(0.1)
        
        if data_source == "Generate Demo Data":
            generator = DataGenerator()
            df = generator.generate_transactions(5000)
        else:
            # File upload would be implemented here
            st.error("File upload not implemented in this demo")
            return
        
        # Step 2: Feature Engineering
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(0.3)
        
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        
        # Step 3: Rule-based Detection
        status_text.text("📋 Evaluating rules...")
        progress_bar.progress(0.5)
        
        rule_engine = RuleEngine()
        rule_scores = rule_engine.evaluate(df)
        
        # Step 4: ML Detection
        status_text.text("🤖 Training ML models...")
        progress_bar.progress(0.7)
        
        ml_engine = MLEngine()
        metrics = ml_engine.train(df)
        rf_scores, iso_scores = ml_engine.predict(df)
        
        # Step 5: Risk Scoring
        status_text.text("⚖️ Computing risk scores...")
        progress_bar.progress(0.9)
        
        risk_scorer = RiskScorer()
        final_scores = risk_scorer.compute_risk_score(rule_scores, rf_scores, iso_scores)
        risk_categories = risk_scorer.categorize_risk(final_scores, 
                                                    config.HIGH_RISK_THRESHOLD,
                                                    config.MEDIUM_RISK_THRESHOLD)
        
        # Add results to dataframe
        df['rule_score'] = rule_scores
        df['ml_score_rf'] = rf_scores
        df['ml_score_iso'] = iso_scores
        df['final_risk_score'] = final_scores
        df['risk_category'] = risk_categories
        
        # Step 6: Save results
        status_text.text("💾 Saving results...")
        progress_bar.progress(1.0)
        
        # Save to session state
        st.session_state.df = df
        st.session_state.ml_engine = ml_engine
        st.session_state.metrics = metrics
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show success message
        high_risk_count = len(df[df['risk_category'] == 'HIGH'])
        st.success(f"✅ Detection pipeline complete! Found {high_risk_count} high-risk transactions.")
        
        # Show key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions Processed", len(df))
        col2.metric("ML Accuracy", f"{metrics['random_forest_accuracy']:.1%}")
        col3.metric("High-Risk Alerts", high_risk_count)
        
    except Exception as e:
        st.error(f"Detection pipeline failed: {e}")
        logger.error(f"Pipeline error: {e}")
    
    finally:
        st.session_state.processing = False

# ==================== TESTING FUNCTIONS ====================

def run_smoke_tests():
    """Run basic smoke tests to validate system functionality"""
    tests = {
        'data_generation': False,
        'feature_engineering': False,
        'rule_engine': False,
        'ml_training': False,
        'storage': False
    }
    
    try:
        # Test data generation
        generator = DataGenerator()
        df = generator.generate_transactions(100)
        tests['data_generation'] = len(df) == 100
        
        # Test feature engineering
        engineer = FeatureEngineer()
        df_features = engineer.transform(df)
        tests['feature_engineering'] = 'amount_log' in df_features.columns
        
        # Test rule engine
        rule_engine = RuleEngine()
        rule_scores = rule_engine.evaluate(df_features)
        tests['rule_engine'] = len(rule_scores) == len(df)
        
        # Test ML training
        ml_engine = MLEngine()
        metrics = ml_engine.train(df_features)
        tests['ml_training'] = ml_engine.is_trained
        
        # Test storage
        test_item = {'alert_id': 'TEST_001', 'priority': 'HIGH'}
        tests['storage'] = storage.save_queue_item(test_item)
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
    
    return tests

# ==================== APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    # Run smoke tests in debug mode
    if config.LOG_LEVEL == "DEBUG":
        st.sidebar.subheader("🧪 System Tests")
        if st.sidebar.button("Run Smoke Tests"):
            test_results = run_smoke_tests()
            for test, passed in test_results.items():
                icon = "✅" if passed else "❌"
                st.sidebar.write(f"{icon} {test}")
    
    # Launch main application
    main()
