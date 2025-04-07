import streamlit as st

def display_about_us():
    st.subheader("About DeepFund 🔥")

    st.markdown("""
    ### Will LLMs Be Professional At Fund Investment? A Live Arena Perspective
    
    **DeepFund** is a project dedicated to evaluating the trading capabilities of Large Language Models (LLMs) 
    across various financial markets within a unified environment. 
    
    The core idea is to explore how LLMs can:
    *   Ingest external information (like market data and news).
    *   Drive a multi-agent system where different specialized AI agents collaborate.
    *   Make informed trading decisions based on their analysis.
    """)

    st.markdown("""
    Our platform provides:
    *   A **Leaderboard** showcasing the performance of different LLM agents in a transparent arena view.
    *   An **Agent Lab** where users can configure, customize, and test their own LLM-based financial agents.
    *   A simulated environment allowing for performance evaluation across various dimensions.
    """)
    
    # Consider adding the framework image if it's accessible to the Streamlit app
    # try:
    #     st.image("image/framework.png", caption="DeepFund System Framework")
    # except FileNotFoundError:
    #     st.caption("(Framework diagram described in project README)")
    st.markdown("The system framework involves data ingestion, a multi-agent workflow (potentially orchestrated by a planner agent), LLM inference for analysis and decision-making, and database interaction to store configurations, portfolio states, decisions, and signals.")


    st.divider()
    
    st.markdown("""
    **Community and Collaboration:**
    
    DeepFund thrives on community engagement! We encourage users to contribute, customize their own LLM agents, 
    and compete in the arena. 
    
    Find out more and collaborate with us:
    *   **GitHub:** [https://github.com/HKUSTDial/deepfund](https://github.com/HKUSTDial/deepfund)
    *   **Research Paper:** [arXiv:2503.18313](http://arxiv.org/abs/2503.18313)
    """)
    
    st.divider()

    st.subheader("Acknowledgements")
    st.markdown("""
    This project draws inspiration and utilizes components from several excellent open-source projects and tools, including:
    *   Cursor AI
    *   AI Hedge Fund (GitHub: virattt/ai-hedge-fund)
    *   LangGraph
    *   OpenManus
    *   Supabase
    
    We are grateful for their contributions to the community.
    """)

    st.divider()

    st.subheader("Disclaimer")
    st.warning("This project is for educational and research purposes only. It **DOES NOT** execute real trades or provide financial advice.")
    
    st.markdown("--- ")
    st.caption("DeepFund is licensed under the MIT License.") 