import streamlit as st
import hmac

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", layout="wide")

# Hide Streamlit's default sidebar elements
st.markdown("""
    <style>
        /* Hide the default Streamlit sidebar navigation */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Hide the hamburger menu if present */
        #MainMenu {
            visibility: hidden;
        }
        
        /* Hide footer */
        footer {
            visibility: hidden;
        }
        
        /* Hide deploy button */
        .stDeployButton {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            correct_password = st.secrets["password"]
        except:
            correct_password = "ford2024"
            
        if hmac.compare_digest(st.session_state["password"], correct_password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Show password input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent.png", width=200)
        st.title("Ford Analytics Portal")
        st.markdown("### Enter the access password")
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            label_visibility="collapsed",
            placeholder="Enter password..."
        )
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Password incorrect")
        
        st.markdown("---")
        st.caption("Contact administrator for access credentials")

    return False

# Check password first
if not check_password():
    st.stop()

# MAIN APPLICATION - Only shown after password authentication
def main():
    # Clear any previous content
    st.empty()
    
    # Create our custom sidebar content
    with st.sidebar:
        # Logo and title in sidebar
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent.png", width=50)
        with col2:
            st.markdown("### Ford Analytics")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("#### Navigation")
        
        # Use buttons instead of radio for better control
        if st.button("📊 Dashboard", use_container_width=True, type="primary" if st.session_state.get('current_page') == 'dashboard' else "secondary"):
            st.session_state.current_page = "dashboard"
            st.switch_page("dashboard.py")
            
        if st.button("💬 SQL Chat", use_container_width=True, type="primary" if st.session_state.get('current_page') == 'sql_chat' else "secondary"):
            st.session_state.current_page = "sql_chat"
            st.switch_page("sql_chat.py")
            
        if st.button("🤖 AI Agent", use_container_width=True, type="primary" if st.session_state.get('current_page') == 'ai_agent' else "secondary"):
            st.session_state.current_page = "ai_agent"
            st.switch_page("ai_agent.py")
        
        st.markdown("---")
        st.markdown("*Use the navigation above to explore different analytics tools*")
    
    # Show welcome content on the main app page
    st.title("🚗 Welcome to Ford Analytics")
    st.markdown("### Select a tool from the sidebar to get started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Dashboard")
        st.markdown("Comprehensive overview of fleet performance, revenue metrics, and customer insights.")
        if st.button("Go to Dashboard", key="dash_btn"):
            st.session_state.current_page = "dashboard"
            st.switch_page("dashboard.py")
    
    with col2:
        st.markdown("#### 💬 SQL Chat")
        st.markdown("Natural language to SQL converter. Ask questions about your data in plain English.")
        if st.button("Go to SQL Chat", key="sql_btn"):
            st.session_state.current_page = "sql_chat"
            st.switch_page("sql_chat.py")
    
    with col3:
        st.markdown("#### 🤖 AI Agent")
        st.markdown("Test business strategies with AI-powered analysis and predictive modeling.")
        if st.button("Go to AI Agent", key="ai_btn"):
            st.session_state.current_page = "ai_agent"
            st.switch_page("ai_agent.py")
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar for quick navigation between tools")

if __name__ == "__main__":
    main()
