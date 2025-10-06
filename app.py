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

# Set session state for current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Sidebar navigation
st.sidebar.title("Ford Analytics")
page = st.sidebar.radio("Navigate to:", 
    ["Dashboard", "SQL Chat", "AI Agent"],
    index=["Dashboard", "SQL Chat", "AI Agent"].index(st.session_state.current_page)
)

# Update session state
if page != st.session_state.current_page:
    st.session_state.current_page = page
    st.rerun()

# Display current page content
if st.session_state.current_page == "Dashboard":
    # Import dashboard content directly
    import dashboard_content
    dashboard_content.show()
    
elif st.session_state.current_page == "SQL Chat":
    # Import SQL chat content directly
    import sql_chat_content
    sql_chat_content.show()
    
elif st.session_state.current_page == "AI Agent":
    # Import AI agent content directly
    import ai_agent_content
    ai_agent_content.show()
