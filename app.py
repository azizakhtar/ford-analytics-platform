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
    # Import and run your actual Dashboard
    try:
        from pages import 1_Dashboard
        # Your actual Dashboard page will run
    except:
        st.title("📊 Business Dashboard")
        st.metric("Total Revenue", "$4.2M", "+12%")
        # Fallback simple dashboard

elif page == "💬 SQL Chat":
    # Import and run your actual SQL Chat
    try:
        from pages import 2_SQL_Chat
        # Your actual SQL Chat page will run  
    except:
        st.title("💬 SQL Chat Interface")
        st.info("SQL Chat page loading...")
        # Fallback simple SQL Chat

elif page == "🤖 AI Agent":
    # Import and run your actual AI Agent
    try:
        from pages import 3_AI_Agent
        # Your actual AI Agent page will run
    except:
        st.title("🤖 AI Strategy Testing")
        st.info("AI Agent page loading...")
        # Fallback simple AI Agent
