import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sqlite3
import time
import logging

# --- 1. SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SpeedyCall Churn Insights",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM CSS ---
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e1b4b, #064e3b); color: #ffffff; }
[data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.9) !important; }
h1, h2, h3 { background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: fadeInDown 1.5s ease-out; }
@keyframes fadeInDown { 0% { opacity: 0; transform: translateY(-30px); } 100% { opacity: 1; transform: translateY(0); } }
.metric-card { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; text-align: center; backdrop-filter: blur(10px); border: 1px solid rgba(74, 222, 128, 0.3); transition: all 0.3s ease; animation: float 4s ease-in-out infinite; }
@keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-10px); } 100% { transform: translateY(0px); } }
.metric-card:hover { transform: scale(1.05) translateY(-5px); border-color: #4ade80; box-shadow: 0px 10px 20px rgba(74, 222, 128, 0.2); animation: none; }
.metric-title { font-size: 1.2rem; color: #e0e0e0; margin-bottom: 5px; }
.metric-value { font-size: 2.8rem; font-weight: 900; color: #4ade80; text-shadow: 0 0 10px rgba(74, 222, 128, 0.4); }
.stButton>button { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 30px; padding: 10px 24px; font-weight: bold; width: 100%; transition: 0.3s; }
.stButton>button:hover { background: linear-gradient(90deg, #8b5cf6, #4ade80); transform: translateY(-2px); color: #ffffff; }
p, li, span { color: #e2e8f0; }
[data-testid="stAlert"] { background-color: rgba(30, 41, 59, 0.6) !important; border-left: 4px solid #4ade80 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# Main Header
st.title("🚀 SpeedyCall Customer Churn Dashboard")
st.markdown("*Real-time interactive insights tracking churn across factors like tenure, contracts, and services.*")
st.markdown("---")

# --- 3. DATABASE ENGINE (SQLite) ---
DB_NAME = 'churn_db.sqlite'
EXCEL_PATH = 'SpeedyCall.xlsx'

def get_db_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customers'")
    if not cursor.fetchone():
        try:
            df = pd.read_excel(EXCEL_PATH)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan)).fillna(0)
            df['customerID'] = df['customerID'].astype(str)
            df.to_sql('customers', conn, if_exists='replace', index=False)
            st.success("✅ Database Initialized!")
        except Exception as e: st.error(f"Migration Error: {e}"); st.stop()
    conn.close()

init_db()

# --- 4. DATA LOADING & ML ---
@st.cache_data
def load_data():
    try:
        conn = get_db_connection(); df = pd.read_sql_query("SELECT * FROM customers", conn); conn.close()
        return df
    except Exception as e: st.error(f"Error: {e}"); return None

@st.cache_resource
def train_model(df):
    try:
        features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'TechSupport', 'SeniorCitizen']
        df_model = df[features + ['Churn']].copy()
        le_dict = {}
        for col in df_model.select_dtypes(include=['object']).columns:
            le = LabelEncoder(); df_model[col] = le.fit_transform(df_model[col].astype(str)); le_dict[col] = le
        X = df_model.drop('Churn', axis=1); y = df_model['Churn']
        rf = RandomForestClassifier(n_estimators=100, random_state=42); rf.fit(X, y)
        return rf, le_dict
    except Exception as e: return None, None

df = load_data()
if df is None or df.empty: st.stop()
model, le_dict = train_model(df)

# --- 5. REFINED DASHBOARD TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💰 Financial Impact", "📉 Service Impact", "⚖️ Policy Impact", "👥 Manage Customers"])

with tab1:
    st.header("🏢 Overview")
    total = len(df); churned = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churned / total) * 100 if total > 0 else 0
    mrr = df[df['Churn'] == 'No']['MonthlyCharges'].sum()
    loss = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="metric-card"><div class="metric-title">👥 Total Customers</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-title">📉 Churn Rate</div><div class="metric-value">{churn_rate:.1f}%</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="metric-title">💵 Active MRR</div><div class="metric-value">${mrr:,.0f}</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card" style="border-left: 4px solid #ef4444;"><div class="metric-title" style="color: #ef4444;">💸 Monthly Loss</div><div class="metric-value" style="color: #ef4444;">${loss:,.0f}</div></div>', unsafe_allow_html=True)
    
    st.plotly_chart(px.histogram(df, x='tenure', color='Churn', title="Loss Risk over Time", color_discrete_sequence=['#3b82f6', '#4ade80'], template="plotly_dark"), use_container_width=True)
    st.success("**Tenure Takeaway:** Highest risk is in the first few months. Passing 12 months heavily reduces churn risk.")
    
    c1, c2 = st.columns(2)
    with c1: 
        df['Dependents Status'] = df['Dependents'].map({'Yes': 'Have Dependents', 'No': 'Does Not Have Dependents'})
        st.plotly_chart(px.pie(df, names='Dependents Status', title="Family Status", hole=0.4, color_discrete_sequence=['#4ade80', '#3b82f6'], template="plotly_dark"), use_container_width=True)
    with c2: 
        df['Senior Text'] = df['SeniorCitizen'].map({1: 'Senior', 0: 'Not Senior'})
        st.plotly_chart(px.bar(df.groupby(['Senior Text', 'Churn']).size().reset_index(name='Count'), x='Senior Text', y='Count', color='Churn', barmode='group', title="Loss by Age Group", template="plotly_dark"), use_container_width=True)
    
    st.markdown("### 💡 Strategic Takeaways")
    k1, k2 = st.columns(2)
    k1.success("**Service Focus:** Quality internet and extra protection matter way more than monthly pricing.")
    k2.success("**Payments:** Manual 'Electronic Checks' push people away. Encourage automated billing.")

with tab2:
    st.header("Financial Impact")
    st.markdown("### 🛠️ Interactive Analysis Filters")
    f_cols = st.columns(2)
    f_contract = f_cols[0].multiselect("Plan Type:", df['Contract'].unique(), default=df['Contract'].unique())
    f_service = f_cols[1].multiselect("Internet Service:", df['InternetService'].unique(), default=df['InternetService'].unique())
    filtered_df = df[(df['Contract'].isin(f_contract)) & (df['InternetService'].isin(f_service))]
    
    col_fin1, col_fin2 = st.columns(2)
    with col_fin1: st.plotly_chart(px.box(filtered_df, x='Churn', y='MonthlyCharges', color='Churn', title="Monthly Charges Distribution", template="plotly_dark"), use_container_width=True)
    with col_fin2: st.plotly_chart(px.box(filtered_df, x='Churn', y='TotalCharges', color='Churn', title="Total Charges Distribution", template="plotly_dark"), use_container_width=True)
    st.info("💡 **Financial Takeaway:** Cost alone isn't the primary driver; contract stability is key.")

with tab3:
    st.header("Service Impact")
    service_cols = ['InternetService', 'TechSupport', 'OnlineSecurity', 'PhoneService', 'StreamingTV', 'StreamingMovies']
    selected_service = st.radio("🔍 Select Service Feature to Analyze:", service_cols, horizontal=True)
    
    insights = {'PhoneService': "Almost everyone (90%) has Phone Service. It doesn't impact churn significantly.", 'InternetService': "Fiber Optic is premium but has the highest loss rate.", 'OnlineSecurity': "Lack of Online Security is a massive churn risk.", 'TechSupport': "Quality Tech Support is a high protective factor.", 'StreamingTV': "Streaming TV acts as a loyalty hook.", 'StreamingMovies': "Bundling movies increases retention."}
    st.info(f"💡 **Insight:** {insights[selected_service]}")
    
    s1, s2 = st.columns([2, 1])
    with s1: 
        s_churn = df.groupby([selected_service, 'Churn']).size().reset_index(name='Count')
        st.plotly_chart(px.bar(s_churn, x=selected_service, y='Count', color='Churn', barmode='group', title=f"Churn by {selected_service}", template="plotly_dark"), use_container_width=True)
    with s2: st.write("Data breakdown"); st.dataframe(s_churn.pivot(index=selected_service, columns='Churn', values='Count').fillna(0).astype(int), use_container_width=True)

with tab4:
    st.header("Policy Impact")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### 📉 Churn Rate by Contract (%)")
        c_rate = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean() * 100).reset_index(name='Rate')
        st.plotly_chart(px.bar(c_rate, x='Contract', y='Rate', color='Contract', text_auto='.1f', title="Churn Rate %", template="plotly_dark"), use_container_width=True)
        st.warning("⚠️ **Strategic Focus:** Focus on converting Month-to-Month users into yearly plans to reduce churn significantly.")
    with col_p2:
        st.markdown("#### 💳 Payment Method Insights (%)")
        pay_df = df.copy(); pay_df['PaymentMethod'] = pay_df['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
        pay_rates = pay_df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean() * 100).reset_index(name='Rate')
        st.plotly_chart(px.bar(pay_rates, x='PaymentMethod', y='Rate', color='PaymentMethod', text_auto='.1f', title="Churn Risk by Payment", template="plotly_dark"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 💰 Pricing & Monthly Payment distribution")
    
    # Sensible Interactions
    p_row1_col1, p_row1_col2 = st.columns([2, 1])
    with p_row1_col1:
        min_m, max_m = float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max())
        charge_range = st.slider("Filter by Monthly Charges ($):", min_m, max_m, (min_m, max_m), step=1.0, key="policy_slider")
    with p_row1_col2:
        show_points = st.radio("Display Mode:", ["Outliers Only", "All Points", "Suspected Outliers"], horizontal=True)
        point_map = {"Outliers Only": "outliers", "All Points": "all", "Suspected Outliers": "suspectedoutliers"}

    policy_df = df[(df['MonthlyCharges'] >= charge_range[0]) & (df['MonthlyCharges'] <= charge_range[1])].copy()
    policy_df['PaymentMethod'] = policy_df['PaymentMethod'].str.replace(' (automatic)', '', regex=False)

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        fig1 = px.box(policy_df, x='Contract', y='MonthlyCharges', color='Churn', 
                      points=point_map[show_points], title="Monthly Charge Distribution by Contract",
                      color_discrete_sequence=['#3b82f6', '#4ade80'], template="plotly_dark",
                      notched=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.info("💡 **Retention Focus:** Long-term contracts (One/Two Year) significantly reduce churn risk. Incentivize month-to-month users to commit to yearly plans.")
        
    with col_box2:
        fig2 = px.box(policy_df, x='PaymentMethod', y='MonthlyCharges', color='Churn',
                      points=point_map[show_points], title="Monthly Charge Distribution by Payment",
                      color_discrete_sequence=['#3b82f6', '#4ade80'], template="plotly_dark",
                      notched=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.success("💡 **Retention Focus:** Automated billing (Credit Card/Bank Transfer) is a major protective factor. Defaulting new customers to automatic payments is advised.")

with tab5:
    st.header("Manage Customers")
    core_cols = ['customerID', 'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
    edit_df = df[core_cols].copy()

    st.markdown("### ➕ Add Customer")
    with st.expander("Open Form"):
        with st.form("insert_form", clear_on_submit=True):
            f_cols = st.columns(3); new_data = {}
            for i, col in enumerate(core_cols):
                if col == 'customerID': continue
                c = f_cols[(i-1)%3]
                if col == 'SeniorCitizen': new_data[col] = c.selectbox(col, [0, 1])
                elif pd.api.types.is_numeric_dtype(df[col]): new_data[col] = c.number_input(col, value=float(df[col].median()))
                else: new_data[col] = c.selectbox(col, df[col].dropna().unique())
            
            if st.form_submit_button("➕ Insert Customer"):
                if new_data['MonthlyCharges'] < 0 or new_data['tenure'] < 0:
                    st.error("⚠️ Validation Error: Charges and Tenure must be positive values.")
                else:
                    try:
                        conn = get_db_connection()
                        ids = pd.to_numeric(df['customerID'], errors='coerce').dropna()
                        new_id = str(int(ids.max() + 1)) if not ids.empty else "1001"
                        new_data['customerID'] = new_id
                        pd.DataFrame([new_data]).to_sql('customers', conn, if_exists='append', index=False)
                        conn.close(); st.cache_data.clear(); st.success(f"✅ Customer {new_id} added!"); time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

    st.markdown("### 🗃️ Management Console")
    s_col, d_col = st.columns([2, 1])
    search_id = s_col.text_input("🔍 Search Database by Customer ID:")
    if search_id: edit_df_v = edit_df[edit_df['customerID'].str.contains(search_id)]
    else: edit_df_v = edit_df
    
    d_col.markdown("<br>", unsafe_allow_html=True)
    d_col.download_button("📥 Download Data", edit_df_v.to_csv(index=False), "churn_database.csv", "text/csv")
    
    edited_grid = st.data_editor(edit_df_v, num_rows="dynamic", use_container_width=True, disabled=['customerID'])
    
    if st.button("💾 Sync Grid to Database", type="primary"):
        st.warning("⚠️ **Safety Check:** This will overwrite the live database records.")
        if st.checkbox("I confirm I want to save these changes"):
            try:
                conn = get_db_connection()
                edited_grid.to_sql('customers', conn, if_exists='replace', index=False)
                conn.close(); st.cache_data.clear(); st.success("✅ Database Synchronized!"); time.sleep(1); st.rerun()
            except Exception as e: st.error(f"Sync failed: {e}")
