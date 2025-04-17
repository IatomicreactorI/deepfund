import streamlit as st
from .utils import load_subpages, get_market_hub_data

def display_markets():
    """主函数，显示市场页面内容"""
    # --- Initialize Session State --- 
    if 'market_view' not in st.session_state:
        st.session_state.market_view = 'hub' # Default view is the main hub
        st.session_state.selected_market_file = None

    # --- Function to change view ---
    def set_market_view(view, file=None):
        st.session_state.market_view = view
        st.session_state.selected_market_file = file

    # --- Title and Initial Description ---
    st.title("📈 Explore Market Sectors 🌐")
    st.markdown("Dive into specific market sectors for detailed analysis, data, and insights. Access to detailed sector pages is a premium feature.")
    
    st.divider()

    # --- Premium Feature Notice --- 
    st.warning("🔒 Premium Feature: Access to detailed market sector analysis requires an active subscription.")
    col1_unlock, col2_unlock, col3_unlock = st.columns([1,2,1])
    with col2_unlock:
        st.button("✨ Unlock Full Market Insights Now!", key="unlock_markets", disabled=True, use_container_width=True)
        st.caption("Upgrade for in-depth data, real-time news feeds (where applicable), and advanced charting tools.")
    
    st.divider()

    # 加载子页面模块
    sub_pages, imports_successful = load_subpages()

    # --- Display Logic based on Session State (Hub Preview Only for non-premium) ---
    if not imports_successful:
        st.warning("Cannot display Markets page content due to import errors.")
        return # Stop execution if imports failed

    # Always show the hub preview regardless of session state if premium is not unlocked
    # In a real app, you'd check a user's subscription status here.
    # For now, we just show the preview and keep subpages inaccessible.
    
    # --- Display Market Hub Preview ---
    st.subheader("📋 Market Sector Previews")
    st.markdown("Unlock premium access to view detailed pages for each sector:")

    # 获取市场数据
    markets_hub = get_market_hub_data()

    # 显示市场卡片
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, market in enumerate(markets_hub):
        with cols[i % 3]:
            with st.container(border=True):
                st.subheader(f"{market['icon']} {market['name']}")
                st.caption(market['desc'])
                # Button remains disabled as it's a premium feature preview
                st.button(
                    f"View {market['name']} Details", 
                    key=f"market_{market['file']}", 
                    disabled=True # Keep disabled
                    # Remove on_click logic for the preview
                    # on_click=set_market_view, 
                    # args=('subpage', market['file']) 
                )

    st.divider()

    # --- Remove Sub-page Logic (as it's premium) ---
    # The following logic would only run if the user has premium access.
    # For this example, we comment it out or remove it.
    """
    elif st.session_state.market_view == 'subpage' and st.session_state.selected_market_file:
        # --- Display Selected Sub-page ---
        selected_file = st.session_state.selected_market_file
        if selected_file in sub_pages:
            # Add a Back button
            if st.button("← Back to Markets Hub", key="back_to_hub"):
                set_market_view('hub') # Go back to the main hub view
                st.rerun() # Rerun immediately to show the hub
            
            # Call the display function from the imported module
            sub_pages[selected_file]() 
        else:
            st.error(f"Could not find the display function for {selected_file}. Returning to hub.")
            set_market_view('hub') # Go back to hub if something went wrong
            st.rerun()
    else:
        # Fallback if state is invalid
        st.error("Invalid market view state. Returning to hub.")
        set_market_view('hub')
        st.rerun()
    """ 