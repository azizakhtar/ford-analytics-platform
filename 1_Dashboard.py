import streamlit as st

st.title("Ford Analytics Dashboard")
st.markdown("Comprehensive overview of fleet performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", "$4.2M", "+12%")
col2.metric("Active Loans", "1,847", "+8%")
col3.metric("Delinquency Rate", "2.3%", "-0.4%")
col4.metric("Customer Satisfaction", "4.2/5", "+0.3")

st.markdown("---")
st.info("Use the SQL Chat page for detailed data queries")
