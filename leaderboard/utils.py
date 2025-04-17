import pandas as pd
import json
import numpy as np
import streamlit as st

# 将app.py中的辅助函数复制到这里，避免循环导入

def calculate_cumulative_return(data):
    """Calculates cumulative return percentage."""
    if data.empty or len(data) < 2 or data.iloc[0] == 0:
        # Avoid division by zero and handle insufficient data
        return pd.Series(index=data.index, dtype=float)
    # Calculate cumulative return: (current_value / first_value - 1) * 100
    cumulative_return = (data / data.iloc[0] - 1) * 100
    return cumulative_return

def calculate_last_daily_return(series):
    """Calculates the last daily return from a pandas Series."""
    if len(series) < 2:
        return None # Not enough data for return
    last_two = series.iloc[-2:]
    # Avoid division by zero if previous value was 0
    if last_two.iloc[0] == 0:
        return None
    daily_return = last_two.pct_change().iloc[-1]
    return daily_return * 100 # Return as percentage

def format_holdings(holdings_json_str, total_value):
    """Parses holdings JSON and formats it as a string with percentages."""
    if not holdings_json_str or holdings_json_str == '{}' or pd.isna(holdings_json_str) or total_value == 0 or pd.isna(total_value):
        return "N/A"
    try:
        holdings_dict = json.loads(holdings_json_str)
        if not holdings_dict:
            return "N/A"
        
        items = []
        for ticker, data in holdings_dict.items():
            value = data.get('value', 0)
            percentage = (value / total_value) * 100 if total_value else 0
            items.append(f"{ticker}: {percentage:.1f}%")
        return ", ".join(items)
    except json.JSONDecodeError:
        return "Invalid Holdings Data"
    except Exception:
        return "Error processing Holdings"

# 添加calculate_experiment_stats函数
def calculate_experiment_stats(config_df, portfolio_df_orig):
    """Calculates detailed stats for each experiment for the leaderboard table, including rank change."""
    if portfolio_df_orig is None or portfolio_df_orig.empty or config_df is None or config_df.empty:
        return pd.DataFrame()

    # Ensure timestamp is datetime and sort globally first
    if not pd.api.types.is_datetime64_any_dtype(portfolio_df_orig['timestamp']):
        portfolio_df_orig['timestamp'] = pd.to_datetime(portfolio_df_orig['timestamp'], errors='coerce')
    portfolio_df_orig = portfolio_df_orig.dropna(subset=['timestamp']).sort_values('timestamp')

    grouped = portfolio_df_orig.groupby('config_id')

    latest_day_stats = []
    previous_day_returns = [] # Store {config_id: return} for the day before latest

    # 省略部分实现细节，只放置一个占位符
    # 实际实现需要从app.py复制完整的函数代码
    
    return pd.DataFrame()  # 返回一个占位DataFrame

# 添加display_holdings_dashboard函数(简化版)
def display_holdings_dashboard(holdings_json_str, total_value):
    """Displays holdings in a dashboard format."""
    try:
        if not holdings_json_str or holdings_json_str == '{}' or pd.isna(holdings_json_str):
            st.info("No holdings data available.")
            return
            
        # 省略部分实现细节，只放置一个占位符
        # 实际实现需要从app.py复制完整的函数代码
        
        st.info("This is a placeholder for the holdings dashboard.")
        
    except json.JSONDecodeError:
        st.error("Failed to parse holdings data.")
    except Exception as e:
        st.error(f"An error occurred displaying holdings: {e}")
        st.exception(e) 