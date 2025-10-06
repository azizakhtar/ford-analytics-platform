
import streamlit as st
import hmac
import hashlib

# ===== PASSWORD PROTECTION =====
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
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

import streamlit as st
import hmac
import hashlib
import os

def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Get password from environment variable or use default
    correct_password = os.getenv("APP_PASSWORD", "ford2024")

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], correct_password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔐 Ford Analytics Portal")
    st.markdown("### Enter the access password to continue")
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

# MAIN APP AFTER PASSWORD
st.set_page_config(page_title="Ford Analytics", page_icon="🚗", layout="wide")
st.title("🚗 Ford Analytics Platform")
st.success("✅ Access granted! Welcome to the analytics platform.")

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

st.markdown("---")
st.markdown("### Your full multi-page app would load here")
st.info("SQL Chat, AI Agent, and Dashboard pages would appear after password authentication")
