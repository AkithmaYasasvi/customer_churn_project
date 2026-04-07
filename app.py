import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# Apply basic page configurations
st.set_page_config(
    page_title="SpeedyCall Churn Insights",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for gorgeous design
st.markdown("""
<style>
/* Gradient background for the main app */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #064e3b);
    color: #ffffff;
}

/* Customizing the sidebar to fit the theme */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.9) !important;
}

/* Animated Headers */
h1, h2, h3 {
    background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeInDown 1.5s ease-out;
}

@keyframes fadeInDown {
    0% { opacity: 0; transform: translateY(-30px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(74, 222, 128, 0.3);
    transition: all 0.3s ease;
    animation: float 4s ease-in-out infinite;
}

.metric-card:hover {
    transform: scale(1.05) translateY(-5px);
    border-color: #4ade80;
    box-shadow: 0px 10px 20px rgba(74, 222, 128, 0.2);
    animation: none;
}

.metric-title {
    font-size: 1.2rem;
    color: #e0e0e0;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 2.8rem;
    font-weight: 900;
    color: #4ade80;
    text-shadow: 0 0 10px rgba(74, 222, 128, 0.4);
    animation: pulse 2s infinite;
}

.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 10px 24px;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease 0s;
    box-shadow: 0px 8px 15px rgba(59, 130, 246, 0.3);
    width: 100%;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #8b5cf6, #4ade80);
    box-shadow: 0px 15px 20px rgba(139, 92, 246, 0.4);
    transform: translateY(-2px);
    color: #ffffff;
}

p, li, span {
    color: #e2e8f0;
}

[data-testid="stAlert"] {
    background-color: rgba(30, 41, 59, 0.6) !important;
    border-left: 4px solid #4ade80 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Main Header
st.title("🚀 SpeedyCall Customer Churn Dashboard")
st.markdown("*Real-time interactive insights tracking churn across factors like tenure, contracts, and services.*")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('SpeedyCall.xlsx')
        # Clean TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Enforce customerID natively starting from 1001
        if 'customerID' not in df.columns or not pd.api.types.is_numeric_dtype(df['customerID']):
            df['customerID'] = range(1001, 1001 + len(df))
        return df
    except Exception as e:
        return None

@st.cache_resource
def train_model(df):
    # Prepare data for simple ML model
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'TechSupport', 'SeniorCitizen']
    
    df_model = df[features + ['Churn']].copy()
    
    # Label Encoding for categorical vars
    le_dict = {}
    for col in df_model.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le
        
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    return rf, le_dict

df = load_data()

if df is None:
    st.error("Dataset 'SpeedyCall.xlsx' not found. Please place it in the same directory.")
    st.stop()

model, le_dict = train_model(df)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💰 Financial Impact", "📉 Service Impact", "⚖️ Policy Impact", "👥 Manage Customers"])

with tab1:
    st.header("🏢 Overview")
    st.write("A simple summary of our customer base and who is leaving.")
    
    # 1. Executive KPI Dashboard
    total_customers = len(df)
    churned_customers = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
    monthly_revenue = df[df['Churn'] == 'No']['MonthlyCharges'].sum()
    revenue_loss = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0s;">
            <div class="metric-title">👥 Total Customers</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.1s;">
            <div class="metric-title">📉 Churn Rate</div>
            <div class="metric-value">{churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.2s;">
            <div class="metric-title">💵 Active MRR</div>
            <div class="metric-value">${monthly_revenue:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.3s; border-left: 4px solid #ef4444;">
            <div class="metric-title" style="color: #ef4444;">💸 Monthly Rev. Loss</div>
            <div class="metric-value" style="color: #ef4444;">${revenue_loss:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Customer Tenure
    st.markdown("### ⏳ Customer Tenure & Loyalty")
    fig_tenure = px.histogram(df, x='tenure', color='Churn', nbins=40,
                        color_discrete_sequence=['#3b82f6', '#4ade80'],
                        title="Customer Loss Risk over Time (Months)",
                        labels={'tenure': 'Tenure (Months)'})
    fig_tenure.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_tenure, use_container_width=True)
    st.success("**Tenure Takeaway:** The highest risk of customers leaving is in the first few months. Passing the 12-month mark heavily reduces churn risk.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Demographic Fact Charts (Simple)
    st.markdown("### 👥 Demographic Fact Breakdown")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        # Pie chart for Dependents
        dep_counts = df['Dependents'].value_counts().reset_index()
        dep_counts.columns = ['Has Dependents?', 'Count']
        fig_dep_pie = px.pie(dep_counts, names='Has Dependents?', values='Count', title="Customer Base by Family Status",
                             color_discrete_sequence=['#4ade80', '#3b82f6'], hole=0.4)
        fig_dep_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_dep_pie, use_container_width=True)

    with col_chart2:
        # Bar chart for Senior Citizens vs Churn
        df['Senior Text'] = df['SeniorCitizen'].map({1: 'Senior', 0: 'Not Senior'})
        senior_churn = df.groupby(['Senior Text', 'Churn']).size().reset_index(name='Count')
        fig_senior_bar = px.bar(senior_churn, x='Senior Text', y='Count', color='Churn', barmode='group',
                                title="Customer Loss by Age Group",
                                color_discrete_sequence=['#3b82f6', '#4ade80'])
        fig_senior_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_senior_bar, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Short & Simple Takeaways
    st.markdown("### 💡 Why Customers Leave (Key Takeaways)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.success("**Service Needs:** Quality internet and extra protection (Tech Support / Online Security) matter way more than monthly pricing.")
        st.success("**Contracts:** Over 50% use month-to-month contracts, which are extremely risky and lead to high churn.")
        
    with col4:
        st.success("**Family Base:** Single customers leave faster. Family households (those with dependents) are much more stable.")
        st.success("**Payments:** Manual payments via 'Electronic Check' push people away. We should encourage automated billing.")


with tab2:
    st.header("Financial Impact")
    st.write("Evaluating how financial factors like Monthly and Total Charges associate with customer churn.")
    
    col_fin1, col_fin2 = st.columns(2)
    
    with col_fin1:
        st.markdown("#### Monthly Charges vs. Churn")
        fig_monthly = px.box(df, x='Churn', y='MonthlyCharges', color='Churn',
                             color_discrete_sequence=['#3b82f6', '#4ade80'],
                             title="Distribution of Monthly Charges")
        fig_monthly.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.markdown("**Observation:** Churned customers tend to have a slightly higher median monthly charge, though cost alone isn't the primary driver.")

    with col_fin2:
        st.markdown("#### Total Charges vs. Churn")
        fig_total = px.box(df, x='Churn', y='TotalCharges', color='Churn',
                           color_discrete_sequence=['#3b82f6', '#4ade80'],
                           title="Distribution of Total Charges")
        fig_total.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_total, use_container_width=True)
        st.markdown("**Observation:** Retained customers generally have higher total charges, driven heavily by their longer tenure.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Financial Takeaway:** Statistical testing reveals *no significant evidence* that churned customers differ strictly based on overall pricing. Qualitative factors like contracts and tech support balance the scales.")
            
with tab3:
    st.header("Service Impact")
    st.write("Explore how core and value-added services affect customer retention.")
    
    st.markdown("### 🔍 Interactive Service Analyzer")
    
    # 1. User Interaction Dropdown
    service_cols = ['InternetService', 'TechSupport', 'OnlineSecurity', 'PhoneService', 'StreamingTV', 'StreamingMovies']
    selected_service = st.selectbox("Select a service feature to analyze:", service_cols)

    # 2. Dynamic Insights
    insights = {
        'PhoneService': "Almost everyone (90%) has Phone Service. Statistically, it doesn't significantly impact whether a customer stays or leaves.",
        'InternetService': "Fiber Optic is our premium service, but it suffers from the highest loss rate. DSL users are much more stable.",
        'OnlineSecurity': "Customers without Online Security leave at a shockingly high rate. It's a massive protective factor.",
        'TechSupport': "Similar to security, lacking Tech Support highly correlates with churn. Providing accessible support retains users.",
        'StreamingTV': "A great value-added service. Customers bundle their entertainment, and its presence strongly influences loyalty.",
        'StreamingMovies': "Much like Streaming TV, bundling movies acts as a hook that influences long-term retention."
    }

    st.info(f"💡 **Insight:** {insights[selected_service]}")

    col_s1, col_s2 = st.columns([2, 1])

    with col_s1:
        # 3. Interactive Chart based on user selection
        service_churn = df.groupby([selected_service, 'Churn']).size().reset_index(name='Count')
        fig_service = px.bar(service_churn, x=selected_service, y='Count', color='Churn', barmode='group',
                      color_discrete_sequence=['#3b82f6', '#4ade80'],
                      title=f"Customer Loss grouped by {selected_service}")
        fig_service.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_service, use_container_width=True)

    with col_s2:
        # 4. Data breakdown
        st.markdown(f"#### Data breakdown")
        pivot_df = service_churn.pivot(index=selected_service, columns='Churn', values='Count').fillna(0).astype(int)
        st.dataframe(pivot_df, use_container_width=True)
    
    st.markdown("---")

with tab4:
    st.header("Policy Impact")
    st.write("Evaluating how organizational policies and billing structures affect customer satisfaction.")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("#### Customer Retention by Contract Type")
        fig_contract = px.bar(df.groupby(['Contract', 'Churn']).size().reset_index(name='Count'),
                              x='Contract', y='Count', color='Churn', barmode='group',
                              color_discrete_sequence=['#3b82f6', '#4ade80'])
        fig_contract.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_contract, use_container_width=True)
        st.success("**Insight:** Month-to-month contracts lack commitment, making them vastly more vulnerable to churn than 1 or 2-year plans.")

    with col_p2:
        st.markdown("#### Customer Retention by Payment Method")
        pay_df = df.copy()
        pay_df['PaymentMethod'] = pay_df['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
        fig_pay = px.bar(pay_df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count'),
                         x='PaymentMethod', y='Count', color='Churn', barmode='group',
                         color_discrete_sequence=['#3b82f6', '#4ade80'])
        fig_pay.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        fig_pay.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pay, use_container_width=True)
        st.success("**Insight:** Manual 'Electronic checks' create payment friction and correlate with significantly higher churn compared to automatic methods.")

    st.markdown("#### 💵 Policy and Cost Convergence")
    st.write("How do Monthly Charges combine with Contract Types to push customers away?")
    
    # Present Monthly Charges in a DIFFERENT way: Grouped by Contract type AND Churn
    fig_policy_cost = px.box(df, x='Contract', y='MonthlyCharges', color='Churn',
                             color_discrete_sequence=['#3b82f6', '#4ade80'])
    fig_policy_cost.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_policy_cost, use_container_width=True)
    st.info("💡 **Cost/Policy Analysis:** When we view Monthly Charges specifically across different contracts, we see a crucial intersection: **Customers on Month-to-Month contracts who face higher monthly charges are highly likely to churn**. The lack of commitment combined with steeper prices acts as a powerful trigger.")

with tab5:
    st.header("Manage Customers")
    st.write("Perform Create, Read, Update, and Delete (CRUD) operations on your customer base. Actions map from this frontend directly to the database backend (`SpeedyCall.xlsx`).")

    # Strictly isolated strictly to the 15 focal variables + ID
    core_cols = [
        'customerID', 'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 
        'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaymentMethod', 'MonthlyCharges', 
        'TotalCharges', 'Churn'
    ]
    
    edit_df = df[core_cols].copy()
    
    st.markdown("### ➕ Add Customer (Insert)")
    with st.expander("Open Form to Insert a New Customer"):
        with st.form("insert_form", clear_on_submit=True):
            cols = st.columns(3)
            new_data = {}
            for i, col in enumerate(core_cols):
                if col == 'customerID':
                    continue
                c = cols[(i - 1) % 3]
                if col == 'SeniorCitizen':
                    mapping = {"Senior citizen": 1, "Not a senior citizen": 0}
                    selected_lbl = c.selectbox(f"{col}", list(mapping.keys()))
                    new_data[col] = mapping[selected_lbl]
                elif pd.api.types.is_numeric_dtype(edit_df[col]):
                    new_data[col] = c.number_input(f"{col}", value=float(edit_df[col].median()))
                else:
                    new_data[col] = c.selectbox(f"{col}", edit_df[col].dropna().unique())
            
            if st.form_submit_button("➕ Insert Customer"):
                # Auto-increment customerID starting safely from 1001
                new_data['customerID'] = edit_df['customerID'].max() + 1 if not edit_df.empty else 1001
                new_row = pd.DataFrame([new_data])
                updated_full_df = pd.concat([edit_df, new_row], ignore_index=True)
                updated_full_df.to_excel('SpeedyCall.xlsx', index=False)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"✅ Customer {new_data['customerID']} successfully inserted into backend database! Reloading...")
                time.sleep(1.5)
                st.rerun()

    st.markdown("### ✏️ Explicit Update via Customer ID")
    update_id = st.number_input("Provide specific Customer ID to securely fetch & update", min_value=1001, step=1)
    
    if update_id in edit_df['customerID'].values:
        cust_row = edit_df[edit_df['customerID'] == update_id].iloc[0]
        with st.form("explicit_update_form", clear_on_submit=False):
            st.info(f"Loaded details for **Customer {update_id}**")
            u_cols = st.columns(3)
            upd_data = {}
            for i, col in enumerate(core_cols):
                if col == 'customerID':
                    continue
                c = u_cols[(i - 1) % 3]
                curr_val = cust_row[col]
                if col == 'SeniorCitizen':
                    mapping = {"Senior citizen": 1, "Not a senior citizen": 0}
                    reverse_mapping = {1: "Senior citizen", 0: "Not a senior citizen"}
                    curr_lbl = reverse_mapping.get(curr_val, "Not a senior citizen")
                    idx = list(mapping.keys()).index(curr_lbl)
                    selected_lbl = c.selectbox(f"Update {col}", list(mapping.keys()), index=idx, key=f"up_{col}")
                    upd_data[col] = mapping[selected_lbl]
                elif pd.api.types.is_numeric_dtype(edit_df[col]):
                    upd_data[col] = c.number_input(f"Update {col}", value=float(curr_val), key=f"up_{col}")
                else:
                    opts = list(edit_df[col].dropna().unique())
                    idx = opts.index(curr_val) if curr_val in opts else 0
                    upd_data[col] = c.selectbox(f"Update {col}", opts, index=idx, key=f"up_{col}")
                    
            if st.form_submit_button("✏️ Push Explicit Update"):
                for col, val in upd_data.items():
                    edit_df.loc[edit_df['customerID'] == update_id, col] = val
                edit_df.to_excel('SpeedyCall.xlsx', index=False)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"✅ Customer {update_id} successfully updated on backend! Reloading...")
                time.sleep(1.5)
                st.rerun()
    else:
        st.warning(f"Customer ID {update_id} currently not found in active database.")

    st.markdown("### 🗃️ Bulk Edit & ❌ Delete Customer")
    st.write("To **Delete**, click the checkbox next to the row(s) and tap the 'Delete' icon or key. You can also mass-edit from this grid.")
    
    edited_df = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True, key="db_editor", disabled=['customerID'])
    
    st.markdown("### 💾 Save Grid Records")
    st.write("Push your inline edits and deletions from the grid directly into the database.")
    if st.button("💾 Save Grid Updates & Deletions", type="primary"):
        try:
            edited_df.to_excel('SpeedyCall.xlsx', index=False)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Database fully synchronized! Edits and deletions applied via backend.")
            time.sleep(1.5)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to sync: {e}")
