fixed_app = '''
import streamlit as st
import hmac
import hashlib

# ===== PASSWORD PROTECTION =====
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Check if password secret is set
    try:
        correct_password = st.secrets["password"]
    except KeyError:
        st.error("🔧 Configuration Error: Password not set in Streamlit secrets")
        st.info("Please add this to your Streamlit Cloud secrets:")
        st.code('''
[secrets]
password = "your_password_here"
''')
        return False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], correct_password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.title("🔐 Ford Analytics Portal")
    st.markdown("### Enter the access password to continue")
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# Check password before showing the app
if not check_password():
    st.stop()

# ===== YOUR EXISTING APP CONTENT =====
st.set_page_config(
    page_title="Ford Fleet Analytics",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Ford Fleet Management Platform")
st.markdown("### Multi-Page Analytics Dashboard")

st.sidebar.success("Select a page from the sidebar")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Vehicles", "1,247")
    st.metric("Active Customers", "8,421")
with col2:
    st.metric("Portfolio Value", "$142M")
    st.metric("AI Insights", "28")
with col3:
    st.metric("Agent Status", "Ready")
    st.metric("Cost Control", "Active")

st.markdown("---")
st.markdown("### Available Pages:")
st.markdown("- **Dashboard**: Overview and metrics")
st.markdown("- **SQL Chat**: Your existing Gemini SQL interface")
st.markdown("- **AI Agent**: Autonomous analytics with cost controls")
'''

with open('app.py', 'w') as f:
    f.write(fixed_app)

print("✅ Created fixed app.py with better error handling")
