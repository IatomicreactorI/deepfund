import streamlit as st
import pandas as pd
import os
import json

def display_analysts_selector():
    """
    显示analysts选择弹窗
    返回所选的analysts列表
    """
    st.subheader("Select Analysts for Your Agent")
    
    # 加载analysts数据
    analysts_df = load_analysts_csv()
    
    # 如果session state中还没有selected_analysts，初始化为空列表
    if 'temp_selected_analysts' not in st.session_state:
        st.session_state.temp_selected_analysts = []
        # 如果已经在agent_config中有选择，则复制过来
        if 'agent_config' in st.session_state and 'selected_analysts' in st.session_state.agent_config:
            st.session_state.temp_selected_analysts = st.session_state.agent_config['selected_analysts'].copy()
    
    selected_analysts = st.session_state.temp_selected_analysts
    
    # 创建用于展示所选数量的指示器
    selection_count = len(selected_analysts)
    
    # 创建标题行，显示所选数量和提交按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        if selection_count > 0:
            st.info(f"You have selected {selection_count} analysts")
        else:
            st.warning("Please select at least one analyst")
    
    
    st.divider()
    
    # 分隔符上方增加一个"全部清除"或"全选"按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Clear All Selections", use_container_width=True):
            st.session_state.temp_selected_analysts = []
            st.rerun()
    
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_analyst_selector = False
            # 取消时清除临时选择
            st.session_state.pop('temp_selected_analysts', None)
            st.rerun()
    
    with col3:
        # 提供一个全选按钮会很有用
        if not analysts_df.empty:
            all_analysts = analysts_df['name'].tolist()
            is_all_selected = set(all_analysts).issubset(set(selected_analysts))
            if is_all_selected:
                button_label = "Deselect All"
            else:
                button_label = "Select All"
                
            if st.button(button_label, use_container_width=True):
                if is_all_selected:
                    st.session_state.temp_selected_analysts = []
                else:
                    st.session_state.temp_selected_analysts = all_analysts.copy()
                st.rerun()
    
    # 创建一个更紧凑的布局来显示分类的analysts
    if not analysts_df.empty:
        # 按类别分组
        analyst_categories = analysts_df.groupby('category')
        
        # 类别颜色映射
        category_colors = {
            "Fundamental Analysis": "rgba(200, 230, 201, 0.4)",  # 浅绿色
            "Technical Analysis": "rgba(179, 229, 252, 0.4)",    # 浅蓝色
            "Market Sentiment": "rgba(255, 224, 178, 0.4)",      # 浅橙色
            "Risk Management": "rgba(225, 190, 231, 0.4)",       # 浅紫色
            "Macroeconomic": "rgba(248, 187, 208, 0.4)"          # 浅粉色
        }
        
        # 获取所有类别
        all_categories = [category for category, _ in analyst_categories]
        
        # 使用expander组件使布局更紧凑
        for category, group in analyst_categories:
            # 获取合适的背景色或使用默认色
            bg_color = category_colors.get(category, "rgba(224, 224, 224, 0.4)")
            
            # 使用expander组件，默认展开
            with st.expander(f"**{category}**", expanded=True):
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 5px; border-radius: 5px;">
                </div>
                """, unsafe_allow_html=True)
                
                # 获取该类别的所有分析师
                analysts_in_category = group['name'].tolist()
                
                # 创建一个4列的布局，使分析师选择更紧凑
                cols_per_row = 4
                cols = st.columns(cols_per_row)
                
                # 填充列
                for i, analyst in enumerate(analysts_in_category):
                    with cols[i % cols_per_row]:
                        analyst_info = group[group['name'] == analyst].iloc[0]
                        is_selected = analyst in selected_analysts
                        
                        # 使用checkbox进行选择
                        if st.checkbox(
                            analyst, 
                            value=is_selected, 
                            key=f"selector_{category}_{analyst}"
                        ):
                            if analyst not in selected_analysts:
                                selected_analysts.append(analyst)
                        else:
                            if analyst in selected_analysts:
                                selected_analysts.remove(analyst)
                        
                        # 显示描述信息
                        st.caption(f"{analyst_info['description']}")
    else:
        # CSV为空时使用演示数据
        st.warning("Could not load analysts from CSV. Using demo analysts instead.")
        
        # 演示分析师数据
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
        
        # 为每个类别创建expander
        for category, analysts in analyst_categories.items():
            # 设置背景色
            if category == "Fundamental Analysis":
                bg_color = "rgba(200, 230, 201, 0.4)"
            elif category == "Technical Analysis":
                bg_color = "rgba(179, 229, 252, 0.4)"
            elif category == "Market Sentiment":
                bg_color = "rgba(255, 224, 178, 0.4)"
            else:  # Risk Management
                bg_color = "rgba(225, 190, 231, 0.4)"
            
            # 使用expander组件，默认展开
            with st.expander(f"**{category}**", expanded=True):
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 5px; border-radius: 5px;">
                </div>
                """, unsafe_allow_html=True)
                
                # 创建一个4列的布局
                cols = st.columns(4)
                
                # 填充列
                for i, analyst in enumerate(analysts):
                    with cols[i % 4]:
                        is_selected = analyst in selected_analysts
                        
                        # 使用checkbox进行选择
                        if st.checkbox(
                            analyst, 
                            value=is_selected, 
                            key=f"demo_selector_{category}_{analyst}"
                        ):
                            if analyst not in selected_analysts:
                                selected_analysts.append(analyst)
                        else:
                            if analyst in selected_analysts:
                                selected_analysts.remove(analyst)
                        
                        # 显示描述信息
                        st.caption(f"Demo {analyst}")

    # 保存选择结果到session_state
    st.session_state.temp_selected_analysts = selected_analysts
    
    # 在弹窗底部再次显示确认按钮，便于用户在选择完成后确认
    st.divider()
    if st.button("✅ Confirm Selection", key="confirm_bottom", use_container_width=True, type="primary"):
        st.session_state.agent_config['selected_analysts'] = selected_analysts.copy()
        st.success("Analysts selection saved!")
        st.session_state.show_analyst_selector = False
        st.rerun()

def load_analysts_csv():
    """从CSV文件加载analysts数据，如果文件不存在或为空则返回空DataFrame"""
    try:
        # 检查文件是否存在且有内容
        if os.path.exists('data/analysts.csv') and os.path.getsize('data/analysts.csv') > 0:
            analysts_df = pd.read_csv('data/analysts.csv')
            return analysts_df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading analysts.csv: {e}")
        return pd.DataFrame() 