import streamlit as st
import hmac

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", layout="wide")

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
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent2.png", width=200)
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
    st.stop()  # Do not continue if check_password is not True.

# Once password is verified, set up the main app
def main():
    # Add logo and title to sidebar
    st.sidebar.markdown(
        """
        <style>
        .sidebar-logo {
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
        }
        .sidebar-logo img {
            margin-right: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Logo and title in sidebar
    col1, col2 = st.sidebar.columns([1, 2])
    with col1:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent2.png", width=50)
    with col2:
        st.markdown("### Ford Analytics")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Dashboard", "SQL Chat", "AI Agent"],
        index=0
    )
    
    # Store current page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()
    
    # Load the appropriate page
    if st.session_state.current_page == "Dashboard":
        import dashboard
        dashboard.main()
    elif st.session_state.current_page == "SQL Chat":
        import sql_chat
        sql_chat.main()
    elif st.session_state.current_page == "AI Agent":
        import ai_agent
        ai_agent.main()

if __name__ == "__main__":
    main()
