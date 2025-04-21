import streamlit as st
import pandas as pd
import os
import json
from .analysts_selector import display_analysts_selector

def display_agent_lab():
    st.title("🧪 Agent Lab - Build & Deploy Your AI Analysts 🤖")

    st.markdown("""
    Welcome to the **DeepFund Agent Lab**, the central hub for creating, customizing, 
    and deploying your own AI-powered financial analysts and trading strategies.
    Leverage the power of Large Language Models (LLMs) to build a customized 
    financial agent tailored to your investment style.
    """)

    st.divider()

    # --- 重置函数 ---
    def reset_agent_config():
        st.session_state.agent_config = {
            'market': None,
            'base_model': None,
            'use_planner': False,
            'selected_analysts': [],
            'risk_preference': 50  # Default to mid-level risk
        }
        # 不重置表单提交标志，保留表单提交后的配置展示

    # --- Initialize Session State for Agent Configuration ---
    if 'agent_config' not in st.session_state:
        reset_agent_config()
    
    # 初始化弹窗控制变量
    if 'show_analyst_selector' not in st.session_state:
        st.session_state.show_analyst_selector = False
        
    # 初始化表单提交标志
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    
    # --- 如果弹窗标志为True，显示弹窗 ---
    if st.session_state.show_analyst_selector:
        display_analysts_selector()
        return  # 弹窗显示时不显示主界面
    
    # --- Agent Creation Section ---
    # 使用更灵活的布局方式放置标题和重置按钮
    header_container = st.container()
    
    # 可以调整此比例来改变标题和按钮的相对宽度
    # 例如 [0.9, 0.1] 会使按钮部分更宽
    title_col, reset_col = header_container.columns([0.92, 0.08])
    
    with title_col:
        st.subheader("🤖 Create Your Custom Agent")
    
    with reset_col:
        # 可以通过CSS调整按钮在列中的位置
        # 这里使用一个容器来放置按钮，便于控制其位置
        button_container = st.container()
        
        # 通过添加垂直空间来调整按钮的垂直位置（如需要）
        # st.write("")  # 添加一些垂直空间，可以取消注释调整
        
        # 使用一个简单的按钮，带有重置图标符号
        if button_container.button("↻", help="Reset all settings to default values", key="reset_button"):
            reset_agent_config()
            st.session_state.form_submitted = False
            st.rerun()
    
    # --- 表单外部的Planner选择 ---
    use_planner = st.checkbox(
        "Use Planner Agent (Orchestrates multiple analysts automatically)", 
        value=st.session_state.agent_config['use_planner'],
        key="planner_checkbox_outside_form"
    )
    
    # --- 如果不使用Planner，显示Analyst选择部分 (表单外) ---
    if not use_planner:
        st.subheader("Select Analysts")
        
        # 显示已选择的分析师数量
        selected_count = len(st.session_state.agent_config['selected_analysts'])
        if selected_count > 0:
            selected_analysts_names = ", ".join(st.session_state.agent_config['selected_analysts'][:3])
            if selected_count > 3:
                selected_analysts_names += f" and {selected_count - 3} more..."
            st.success(f"Selected {selected_count} analysts: {selected_analysts_names}")
        else:
            st.warning("No analysts selected yet. Click the button below to select.")
        
        # 创建一个按钮来打开分析师选择弹窗 (表单外)
        if st.button("📋 Open Analysts Selector"):
            st.session_state.show_analyst_selector = True
            st.rerun()
    else:
        st.info("The Planner agent will automatically select and coordinate appropriate analysts based on market conditions and your risk preference.")
    
    # 添加一个分隔符以增强视觉区分
    st.divider()
    
    # --- 其余配置项放在表单内 ---
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
            if not use_planner and not st.session_state.agent_config['selected_analysts']:
                st.error("Please select at least one analyst or enable the Planner.")
                return
                
            # 保存当前配置（用于显示配置摘要）
            temp_config = {
                'market': selected_market,
                'base_model': selected_model,
                'use_planner': use_planner,
                'risk_preference': risk_preference,
                'selected_analysts': st.session_state.agent_config['selected_analysts']
            }
            
            # 如果使用planner，清空已选分析师
            if use_planner:
                temp_config['selected_analysts'] = []
            
            # 设置表单已提交标志和保存描述
            st.session_state.form_submitted = True
            st.session_state.last_risk_description = risk_description
            st.session_state.last_config = temp_config  # 保存配置以供显示
            
            # 重置配置为默认值
            reset_agent_config()
    
    # --- 表单提交后的显示内容（在表单外部）---
    if st.session_state.form_submitted and hasattr(st.session_state, 'last_config'):
        st.success(f"Agent configured successfully! Your agent will trade in the {st.session_state.last_config['market']} market.")
        
        # Display configuration summary
        st.subheader("Agent Configuration")
        
        # 获取保存的数据
        last_config = st.session_state.last_config
        selected_market = last_config['market']
        selected_model = last_config['base_model']
        use_planner = last_config['use_planner']
        risk_preference = last_config['risk_preference']
        risk_description = st.session_state.last_risk_description
        
        # Format the JSON for display
        config_display = {
            "Market": selected_market,
            "Base Model": selected_model,
            "Using Planner": "Yes" if use_planner else "No",
            "Risk Profile": f"{risk_preference}% ({risk_description.split(':')[0]})"
        }
        
        if not use_planner:
            config_display["Selected Analysts"] = last_config['selected_analysts']
            
        # Show formatted configuration
        st.json(config_display)
        
        # Option to download configuration (表单外部)
        config_json = json.dumps(last_config, indent=2)
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
                    st.rerun()
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