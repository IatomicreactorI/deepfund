import streamlit as st
import os

# --- Import Sub-page Modules --- 
# Note: This might fail if __init__.py was not created successfully
try:
    from market_pages import cs2_skins
    from market_pages import crypto
    from market_pages import gold
    from market_pages import oil
    from market_pages import renewables
    from market_pages import us_stocks_full
    sub_pages = {
        'cs2_skins.py': cs2_skins.display_cs2_skins_market,
        'crypto.py': crypto.display_crypto_market,
        'gold.py': gold.display_gold_market,
        'oil.py': oil.display_oil_market,
        'renewables.py': renewables.display_renewables_market,
        'us_stocks_full.py': us_stocks_full.display_us_stocks_full_market
    }
    imports_successful = True
except ImportError as e:
    st.error(f"Failed to import market sub-pages. Ensure 'market_pages/__init__.py' exists. Error: {e}")
    imports_successful = False
    sub_pages = {}
# -------------------------------

def display_markets():
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

    # Define market sectors (Ensure keys match the sub_pages dict)
    markets_hub = [
        {'name': 'Gold', 'icon': '🧈', 'file': 'gold.py', 'desc': 'Precious metals analysis and forecasts.'},
        {'name': 'Oil', 'icon': '🛢️', 'file': 'oil.py', 'desc': 'Energy market trends and crude oil data.'},
        {'name': 'Cryptocurrencies', 'icon': '₿', 'file': 'crypto.py', 'desc': 'Digital asset insights and blockchain news.'},
        {'name': 'US Stocks', 'icon': '📈', 'file': 'us_stocks_full.py', 'desc': 'Comprehensive US stock market data.'},
        {'name': 'Renewable Energy', 'icon': '🔋', 'file': 'renewables.py', 'desc': 'Green energy sector performance.'},
        {'name': 'CS2 Skins Trading', 'icon': '🌈', 'file': 'cs2_skins.py', 'desc': 'Virtual item market analysis (CS2).'}
    ]

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

# Ensure no other functions are defined in this file 