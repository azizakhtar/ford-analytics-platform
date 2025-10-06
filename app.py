import streamlit as st
import hmac

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
    st.switch_page("1_Dashboard.py")
elif page == "💬 SQL Chat":
    st.switch_page("2_SQL_Chat.py")
elif page == "🤖 AI Agent":
    st.switch_page("3_AI_Agent.py")
