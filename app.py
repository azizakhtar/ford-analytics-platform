import streamlit as st
import hmac

def check_password():
    try:
        correct_password = st.secrets["password"]
    except:
        st.error("Please set password in Streamlit secrets")
        return False
    
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
        
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if hmac.compare_digest(pwd, correct_password):
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False

if not check_password():
    st.stop()

# Main app
st.set_page_config(page_title="Ford Analytics", page_icon="🚗", layout="wide")
st.title("🚗 Ford Analytics Platform")
st.success("Access granted!")

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
st.markdown("### Available Pages:")
st.markdown("- **Dashboard**: Overview and metrics")
st.markdown("- **SQL Chat**: Natural language to SQL")
st.markdown("- **AI Agent**: Business strategy testing")
