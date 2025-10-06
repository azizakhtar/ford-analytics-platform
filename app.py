import streamlit as st
import hmac

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", layout="wide")

def check_password():
    try:
        correct_password = st.secrets["password"]
    except:
        correct_password = "ford2024"
    
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
        
    st.title("Ford Analytics Portal")
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

# Add logo using columns (more reliable than CSS)
col1, col2 = st.sidebar.columns([1, 3])
with col1:
    st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent2.png", width=80)
with col2:
    st.title("Ford Analytics")

# MANUAL NAVIGATION using session state instead of switch_page
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", 
    ["Dashboard", "SQL Chat", "AI Agent"])

# Set the page in session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

if page != st.session_state.current_page:
    st.session_state.current_page = page
    st.rerun()

# Load the appropriate page based on session state
if st.session_state.current_page == "Dashboard":
    # Import and run dashboard
    import importlib
    import sys
    sys.path.append('.')
    from 1_Dashboard import main as dashboard_main
    dashboard_main()
    
elif st.session_state.current_page == "SQL Chat":
    # Import and run SQL Chat
    import importlib
    import sys
    sys.path.append('.')
    from 2_SQL_Chat import main as sql_main
    sql_main()
    
elif st.session_state.current_page == "AI Agent":
    # Import and run AI Agent
    import importlib
    import sys
    sys.path.append('.')
    from 3_AI_Agent import main as ai_main
    ai_main()
