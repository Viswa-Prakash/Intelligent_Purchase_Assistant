import streamlit as st
from langchain_core.messages import HumanMessage
from ReAct_Agent import ecommerce_agent

st.set_page_config(page_title="Intelligent Purchase Assistant", page_icon=":robot:")

st.title("Intelligent Purchase Assistant")


st.markdown("""
            Find me the best price for ‘Sony WH-1000XM5’ headphones online. 
            What’s the average user rating, and if I buy from the US for €350, 
            how much is the import tax to Germany?
            """)

with st.form("user_form"):
    user_query = st.text_area("Enter your query here :", height=60)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Agent is analyzing..."):
        output = ecommerce_agent.invoke({"messages": [HumanMessage(content=user_query)]})
        # Show **only the last agent message** (the Final Answer)
        final_message = None
        for msg in reversed(output["messages"]):
            content = getattr(msg, "content", "")
            if "final answer" in content.lower():
                final_message = content
                break
        if not final_message:
            # fallback: just show last assistant/system message
            last = output["messages"][-1]
            final_message = getattr(last, "content", str(last))
        st.markdown("**Here’s a clear summary of your requests and answers:**\n\n" + final_message)
