import streamlit as st
import pandas as pd
import os
import json

def display_agent_lab():
    st.title("🧪 Agent Lab - Build & Deploy Your AI Analysts 🤖")

    st.markdown("""
    Welcome to the **DeepFund Agent Lab**, the central hub for creating, customizing, 
    and deploying your own AI-powered financial analysts and trading strategies.
    Leverage the power of Large Language Models (LLMs) to build a customized 
    financial agent tailored to your investment style.
    """)

    st.divider()

    # --- Initialize Session State for Agent Configuration ---
    if 'agent_config' not in st.session_state:
        st.session_state.agent_config = {
            'market': None,
            'base_model': None,
            'use_planner': False,
            'selected_analysts': [],
            'risk_preference': 50  # Default to mid-level risk
        }
    
    # --- Agent Creation Form ---
    st.subheader("🤖 Create Your Custom Agent")
    
    with st.form("agent_creation_form"):
        # 1. Market Selection
        market_options = [
            "US Stocks", 
            "Cryptocurrencies", 
            "Gold", 
            "Oil", 
            "Renewable Energy", 
            "CS2 Skins"
        ]
        selected_market = st.selectbox(
            "Select Market to Trade:", 
            options=market_options,
            index=0 if st.session_state.agent_config['market'] is None 
                  else market_options.index(st.session_state.agent_config['market'])
        )
        
        # 2. Base Model Selection
        model_options = [
            "GPT-4", 
            "Claude 3 Opus", 
            "Claude 3 Sonnet", 
            "Gemini Pro", 
            "Llama 3 70B", 
            "Mixtral 8x7B"
        ]
        selected_model = st.selectbox(
            "Select Base LLM:", 
            options=model_options,
            index=0 if st.session_state.agent_config['base_model'] is None 
                  else model_options.index(st.session_state.agent_config['base_model'])
        )
        
        # 3. Planner Selection
        use_planner = st.checkbox(
            "Use Planner Agent (Orchestrates multiple analysts automatically)", 
            value=st.session_state.agent_config['use_planner']
        )
        
        # Show different options based on planner selection
        if use_planner:
            st.info("The Planner agent will automatically select and coordinate appropriate analysts based on market conditions and your risk preference.")
        else:
            # 4. Analyst Selection (only if not using planner)
            st.subheader("Select Analysts")
            
            # Load analysts from CSV
            analysts_df = load_analysts_csv()
            
            if not analysts_df.empty:
                # Group analysts by category
                analyst_categories = analysts_df.groupby('category')
                
                # Display analysts by category with different colors
                selected_analysts = []
                
                # Category color map
                category_colors = {
                    "Fundamental Analysis": "rgba(200, 230, 201, 0.5)",  # Light Green
                    "Technical Analysis": "rgba(179, 229, 252, 0.5)",    # Light Blue
                    "Market Sentiment": "rgba(255, 224, 178, 0.5)",      # Light Orange
                    "Risk Management": "rgba(225, 190, 231, 0.5)",       # Light Purple
                    "Macroeconomic": "rgba(248, 187, 208, 0.5)"          # Light Pink
                }
                
                # Process each category
                for category, group in analyst_categories:
                    # Get the appropriate color or use a default if category not in map
                    bg_color = category_colors.get(category, "rgba(224, 224, 224, 0.5)")  # Default light gray
                    
                    # Container with styled background
                    container_style = f"background-color: {bg_color}; border-radius: 5px; padding: 10px; margin-bottom: 10px;"
                    
                    st.markdown(f"""
                    <div style="{container_style}">
                    <h4>{category}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a row of columns for each group of 3 analysts
                    analysts_in_category = group['name'].tolist()
                    
                    # Create three columns per row
                    for i in range(0, len(analysts_in_category), 3):
                        cols = st.columns(3)
                        
                        # Fill the columns with analysts (up to 3 per row)
                        for j in range(3):
                            if i + j < len(analysts_in_category):
                                analyst = analysts_in_category[i + j]
                                analyst_info = group[group['name'] == analyst].iloc[0]
                                
                                with cols[j]:
                                    # Use checkbox for selection
                                    is_selected = analyst in st.session_state.agent_config['selected_analysts']
                                    if st.checkbox(analyst, value=is_selected, key=f"analyst_{category}_{analyst}"):
                                        selected_analysts.append(analyst)
                                    
                                    # Show tooltip with description on hover
                                    st.caption(f"ℹ️ {analyst_info['description']}")
            else:
                # Fallback to demo analysts if CSV is empty or can't be loaded
                st.warning("Could not load analysts from CSV. Using demo analysts instead.")
                
                # Demo analyst categories (same as before)
                analyst_categories = {
                    "Fundamental Analysis": [
                        "Financial Statement Analyst", 
                        "Valuation Expert", 
                        "Industry Specialist"
                    ],
                    "Technical Analysis": [
                        "Chart Pattern Analyst", 
                        "Momentum Tracker", 
                        "Volatility Expert"
                    ],
                    "Market Sentiment": [
                        "News Sentiment Analyst", 
                        "Social Media Monitor", 
                        "Earnings Call Specialist"
                    ],
                    "Risk Management": [
                        "Portfolio Risk Analyst", 
                        "Hedging Strategist", 
                        "Drawdown Protector"
                    ]
                }
                
                # Display demo analysts by category
                selected_analysts = []
                
                for category, analysts in analyst_categories.items():
                    # Use different background colors for different categories
                    if category == "Fundamental Analysis":
                        container_style = "background-color: rgba(200, 230, 201, 0.5); border-radius: 5px; padding: 10px;"
                    elif category == "Technical Analysis":
                        container_style = "background-color: rgba(179, 229, 252, 0.5); border-radius: 5px; padding: 10px;"
                    elif category == "Market Sentiment":
                        container_style = "background-color: rgba(255, 224, 178, 0.5); border-radius: 5px; padding: 10px;"
                    else:  # Risk Management
                        container_style = "background-color: rgba(225, 190, 231, 0.5); border-radius: 5px; padding: 10px;"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="{container_style}">
                        <h4>{category}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create checkboxes for analysts in this category
                        category_selections = []
                        cols = st.columns(3)
                        for i, analyst in enumerate(analysts):
                            with cols[i % 3]:
                                is_selected = analyst in st.session_state.agent_config['selected_analysts']
                                if st.checkbox(analyst, value=is_selected, key=f"analyst_{category}_{analyst}"):
                                    category_selections.append(analyst)
                                
                                # Add a generic description for demo analysts
                                st.caption(f"ℹ️ Demo {analyst}")
                        
                        selected_analysts.extend(category_selections)
            
            # Display a counter of selected analysts
            if selected_analysts:
                st.info(f"You have selected {len(selected_analysts)} analysts.")
            else:
                st.warning("Please select at least one analyst or enable the Planner.")
        
        # 5. Risk Preference Slider
        risk_preference = st.slider(
            "Risk Preference", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.agent_config['risk_preference'],
            step=5,
            format="%d%%",
            help="0% = Very Conservative, 50% = Balanced, 100% = Aggressive"
        )
        
        # Display the risk level description based on slider value
        if risk_preference < 25:
            risk_description = "Very Conservative: Focus on capital preservation with minimal volatility."
        elif risk_preference < 50:
            risk_description = "Conservative: Prefer stable returns with lower risk."
        elif risk_preference < 75:
            risk_description = "Balanced: Moderate risk for moderate returns."
        elif risk_preference < 90:
            risk_description = "Growth-Oriented: Accept higher volatility for greater returns."
        else:
            risk_description = "Aggressive: Maximize returns, willing to accept significant volatility."
            
        st.caption(risk_description)
        
        # Submit button
        col1, col2 = st.columns([1, 1])
        with col1:
            st.caption("Click 'Create Agent' to configure your AI analyst with the selected options.")
        with col2:
            submitted = st.form_submit_button("Create Agent", use_container_width=True, type="primary")
        
        if submitted:
            # Validation
            if not use_planner and not selected_analysts:
                st.error("Please select at least one analyst or enable the Planner.")
                return
                
            # Save the configuration
            st.session_state.agent_config = {
                'market': selected_market,
                'base_model': selected_model,
                'use_planner': use_planner,
                'selected_analysts': selected_analysts if not use_planner else [],
                'risk_preference': risk_preference
            }
            
            st.success(f"Agent configured successfully! Your agent will trade in the {selected_market} market.")
            
            # Display configuration summary
            st.subheader("Agent Configuration")
            
            # Format the JSON for display
            config_display = {
                "Market": selected_market,
                "Base Model": selected_model,
                "Using Planner": "Yes" if use_planner else "No",
                "Risk Profile": f"{risk_preference}% ({risk_description.split(':')[0]})"
            }
            
            if not use_planner:
                config_display["Selected Analysts"] = selected_analysts
                
            # Show formatted configuration
            st.json(config_display)
            
            # Option to download configuration
            config_json = json.dumps(st.session_state.agent_config, indent=2)
            st.download_button(
                label="Export Agent Config (JSON)",
                data=config_json,
                file_name=f"{selected_market.replace(' ', '_').lower()}_agent_config.json",
                mime='application/json',
            )
    
    st.divider()
    
    # --- Agent Management Section ---
    st.subheader("🛠️ Manage Your Agents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Import Existing Configuration**")
        config_file = st.file_uploader("Import Agent Config (JSON)", type=['json'], accept_multiple_files=False)
        if config_file is not None:
            try:
                imported_config = json.loads(config_file.getvalue().decode("utf-8"))
                if st.button("Load Imported Configuration"):
                    st.session_state.agent_config = imported_config
                    st.success("Configuration loaded successfully!")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading configuration: {e}")
    
    with col2:
        st.info("**Test Your Agent**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Run Backtest", disabled=not st.session_state.agent_config['market'], use_container_width=True):
                st.info("Backtesting functionality coming soon...")
        
        with col_b:
            if st.button("Deploy for Paper Trading", disabled=not st.session_state.agent_config['market'], use_container_width=True):
                st.info("Paper trading functionality coming soon...")

    st.divider()
    
    # --- Information Section ---
    st.markdown("**Note:** The Agent Lab is currently in beta. More features like backtesting, live paper trading, and detailed performance analytics are under development.")


def load_analysts_csv():
    """Load analysts from CSV file or return empty DataFrame if file doesn't exist or is empty."""
    try:
        # Check if the file exists and has content
        if os.path.exists('data/analysts.csv') and os.path.getsize('data/analysts.csv') > 0:
            analysts_df = pd.read_csv('data/analysts.csv')
            return analysts_df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading analysts.csv: {e}")
        return pd.DataFrame() 