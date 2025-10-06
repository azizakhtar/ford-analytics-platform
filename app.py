import streamlit as st
import hmac
import os

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", page_icon="🚗", layout="wide")

def check_password():
    try:
        correct_password = st.secrets["password"]
    except:
        correct_password = "ford2024"
    
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
        
    st.title("🔐 Ford Analytics Portal")
    st.markdown("### Enter the access password")
    pwd = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Login"):
        if hmac.compare_digest(pwd, correct_password):
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False

if not check_password():
    st.stop()

# MANUAL NAVIGATION
st.sidebar.title("🚗 Ford Analytics")
page = st.sidebar.radio("Navigate to:", 
    ["📊 Dashboard", "💬 SQL Chat", "🤖 AI Agent"])

if page == "📊 Dashboard":
    st.title("📊 Business Dashboard")
    st.success("✅ Access granted! Welcome to Ford Analytics.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", "$4.2M", "+12%")
        st.metric("Active Customers", "8,421")
    with col2:
        st.metric("Active Loans", "1,847", "+8%")
        st.metric("Portfolio Value", "$142M")
    with col3:
        st.metric("Delinquency Rate", "2.3%", "-0.4%")
        st.metric("AI Insights", "28")

elif page == "💬 SQL Chat":
    st.title("💬 SQL Chat Interface")
    st.info("Natural language to SQL conversion")
    
    # Add your SQL Chat functionality here
    query = st.text_area("Enter your data question:")
    if st.button("Run Query"):
        st.success(f"Processing: {query}")
        # Your SQL Chat code would go here

elif page == "🤖 AI Agent":
    st.title("🤖 AI Strategy Testing") 
    st.info("AI-powered business analysis")
    
    # Add your AI Agent functionality here
    if st.button("Run AI Analysis"):
        st.success("AI analysis complete!")
        # Your AI Agent code would go here
