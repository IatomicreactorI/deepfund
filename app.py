import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
from streamlit_option_menu import option_menu
import json # Needed for parsing holdings
import numpy as np

from agent_lab import display_agent_lab  
from about_us import display_about_us
from market_pages import display_markets  # 从重构后的market_pages模块导入
from reports import display_reports
from community import display_community  # 导入新的community模块

# leaderboard模块暂时不导入，因为我们还没有完成所有功能的迁移
# 在完成迁移后，可以添加: from leaderboard import display_leaderboard

st.set_page_config(
    page_title="DeepFund",
    page_icon="image/logopure.svg",
    layout="wide"
)

# --- Marquee CSS (Updated for Fixed Positioning and Padding) ---
marquee_css = """
<style>
/* Hide Streamlit's default header */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Main content padding to prevent overlap with fixed banners */
/* Try targeting the main block container directly */
.main .block-container {
    padding-top: 0px !important;  /* Reduced top padding */
    padding-bottom: 50px !important; /* Adjust if bottom banner height changes */
}
/* Fallback/Alternative targets */
div[data-testid="stAppViewContainer"] > section > div[data-testid="stVerticalBlock"] {
    padding-top: 0px !important; /* Reduced top padding */
    padding-bottom: 50px !important;
}
div.block-container {
    padding-top: 0px !important; /* Reduced top padding */
    padding-bottom: 50px !important;
}


.marquee-container {
    width: 100%;
    overflow: hidden;
    background-color: #f0f2f6;
    padding: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    position: fixed;
    left: 0;
    z-index: 9999; /* Greatly increased z-index */
}

.top-banner {
    top: 0;
    border-bottom: 1px solid #e0e0e0;
}

.bottom-banner {
    bottom: 0;
    border-top: 1px solid #ccc;
    background-color: #e9ecef;
}

.marquee-content {
    display: inline-block;
    white-space: nowrap;
    /* Slower speed - increased duration */
    animation: marquee 120s linear infinite; /* Increased duration for slower scroll */
    /* padding-left: 100%; Removed for immediate full display */
    font-size: 15px;
    color: #495057;
}

/* Reverse direction specifically for the top banner */
.top-banner .marquee-content {
    animation-direction: reverse; /* Scroll left-to-right */
}


.marquee-container:hover .marquee-content {
    animation-play-state: paused;
}

.marquee-content span.item-separator {
    margin: 0 8px; /* Reduced margin for closer spacing */
    color: #adb5bd;
    font-weight: bold;
}

.marquee-content span.exp-name {
    font-weight: bold;
    color: #007bff;
}

.positive-return {
    color: #28a745;
    font-weight: bold;
}

.negative-return {
    color: #dc3545;
    font-weight: bold;
}

@keyframes marquee {
    0%   { transform: translateX(0); }
    /* Adjusted for seamless loop with duplicated content */
    100% { transform: translateX(-50%); }
}

/* Reduce general vertical spacing for components */
div[data-testid="stVerticalBlock"] > *,
div[data-testid="stVerticalBlock"] {
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}

/* Reduce paragraph spacing specifically */
p {
    margin-top: 0.1rem !important;
    margin-bottom: 0.5rem !important; /* Allow a bit more space below paragraphs */
    line-height: 1.3; /* Slightly reduce line height within paragraphs */
}

/* Reduce space around dividers */
hr[data-testid="stDivider"] {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

/* Reduce space below title */
h1[data-testid="stHeading"] {
    margin-bottom: 0.3rem !important;
}

/* Reduce space below subheaders */
h3[data-testid="stHeading"] {
     margin-top: 0.8rem !important; /* Keep slightly more space above subheaders */
     margin-bottom: 0.3rem !important;
}

</style>
"""

# --- Helper Function for Return Calculation ---
def calculate_cumulative_return(data):
    """Calculates cumulative return percentage."""
    if data.empty or len(data) < 2 or data.iloc[0] == 0:
        # Avoid division by zero and handle insufficient data
        return pd.Series(index=data.index, dtype=float)
    # Calculate cumulative return: (current_value / first_value - 1) * 100
    cumulative_return = (data / data.iloc[0] - 1) * 100
    return cumulative_return

# --- Helper Function for Last Daily Return ---
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

# --- Helper Function to Parse Holdings --- 
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

# --- Function to Calculate Stats for Leaderboard Table ---
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

    # --- First Pass: Calculate latest stats and identify previous day's return ---
    for config_id, group in grouped:
        if group.empty:
            continue

        group_sorted = group.sort_values('timestamp') # Already sorted globally, but good practice

        # Latest day calculations
        latest_row = group_sorted.iloc[-1]
        latest_date = latest_row['timestamp']
        latest_total_value = latest_row['total_value']
        latest_holdings_str = latest_row['holdings']
        start_date = group_sorted['timestamp'].min().strftime('%Y-%m-%d')
        daily_return_pct = calculate_last_daily_return(group_sorted['total_value'])
        total_cumulative_return_series = calculate_cumulative_return(group_sorted['total_value'])
        total_return_pct_latest = total_cumulative_return_series.iloc[-1] if not total_cumulative_return_series.empty else None

        # Get Config Info
        config_info = config_df[config_df['id'] == config_id].iloc[0] if config_id in config_df['id'].values else None
        exp_name = config_info['exp_name'] if config_info is not None else f"Unknown ({config_id[:6]}...)"
        llm_model = config_info['llm_model'] if config_info is not None else "N/A"

        latest_day_stats.append({
            'config_id': config_id,
            'LLM Model': llm_model,
            'Start Date': start_date,
            'End Date': latest_date, # Add End Date (will format later)
            'Decision Accuracy (%)': 'N/A', # Placeholder
            'Daily Return (%)': daily_return_pct,
            'Analyst Portfolio': 'N/A', # Placeholder
            'Total Return (%)': total_return_pct_latest,
            'Current Total Value ($)': latest_total_value,
        })

        # Previous day calculation (if exists)
        if len(group_sorted) >= 2:
            previous_day_data = group_sorted.iloc[:-1]['total_value']
            if not previous_day_data.empty:
                 previous_cumulative_return_series = calculate_cumulative_return(previous_day_data)
                 total_return_pct_previous = previous_cumulative_return_series.iloc[-1] if not previous_cumulative_return_series.empty else None
                 if pd.notna(total_return_pct_previous):
                     previous_day_returns.append({
                         'config_id': config_id,
                         'Previous Total Return (%)': total_return_pct_previous
                     })


    if not latest_day_stats:
        return pd.DataFrame()

    # --- Second Pass: Calculate Ranks and Rank Change ---
    latest_stats_df = pd.DataFrame(latest_day_stats)

    # Calculate current rank
    latest_stats_df = latest_stats_df.sort_values(by='Total Return (%)', ascending=False, na_position='last')
    latest_stats_df['Rank'] = range(1, len(latest_stats_df) + 1)

    # Calculate previous rank (if data exists)
    if previous_day_returns:
        previous_stats_df = pd.DataFrame(previous_day_returns)
        previous_stats_df = previous_stats_df.sort_values(by='Previous Total Return (%)', ascending=False, na_position='last')
        previous_stats_df['Previous Rank'] = range(1, len(previous_stats_df) + 1)

        # Merge previous rank into latest stats
        final_stats_df = pd.merge(latest_stats_df, previous_stats_df[['config_id', 'Previous Rank']], on='config_id', how='left')

        # Calculate Rank Change
        final_stats_df['Rank Change Raw'] = final_stats_df['Previous Rank'] - final_stats_df['Rank']

        # Format Rank Change
        def format_rank_change(row):
            if pd.isna(row['Previous Rank']):
                return 'New'
            change = row['Rank Change Raw']
            if pd.isna(change): # Should not happen if Previous Rank exists, but safety check
                return 'N/A'
            elif change == 0:
                return '0'
            elif change > 0:
                return f"+{int(change)}"
            else:
                return f"{int(change)}"

        final_stats_df['Rank Change'] = final_stats_df.apply(format_rank_change, axis=1)
        final_stats_df = final_stats_df.drop(columns=['Rank Change Raw', 'Previous Rank']) # Clean up temp columns

    else:
        # No previous data available for any experiment
        final_stats_df = latest_stats_df
        final_stats_df['Rank Change'] = 'New'


    # Format End Date
    final_stats_df['End Date'] = final_stats_df['End Date'].dt.strftime('%Y-%m-%d')

    # --- Defer numeric formatting to display function ---

    # Select and reorder columns for final display
    final_columns = [
        'Rank',
        'Rank Change', # Now calculated
        'LLM Model',
        'Start Date',
        'End Date', # Added
        'Decision Accuracy (%)',
        'Daily Return (%)',
        'Analyst Portfolio',
        'Total Return (%)',
        'Current Total Value ($)',
    ]
    # Ensure only existing columns are selected and in order
    final_stats_df = final_stats_df[[col for col in final_columns if col in final_stats_df.columns]]
    return final_stats_df

# --- Data Calculation for Banners (Keep existing, maybe refactor later) --- 
def get_banner_data(config_df, portfolio_df_orig):
    """Calculates data for top (latest value) and bottom (latest daily return) banners."""
    latest_values_str = "<span>No portfolio data available</span>"
    latest_returns_str = "<span>No portfolio data available</span>"

    if portfolio_df_orig is None or portfolio_df_orig.empty or config_df is None or config_df.empty:
        return latest_values_str, latest_returns_str

    try:
        # Ensure 'timestamp' is datetime for sorting/comparison
        if not pd.api.types.is_datetime64_any_dtype(portfolio_df_orig['timestamp']):
             portfolio_df_orig['timestamp'] = pd.to_datetime(portfolio_df_orig['timestamp'])

        # Get latest entry for each config_id based on timestamp
        # Use idxmax on timestamp after grouping by config_id
        latest_indices = portfolio_df_orig.loc[portfolio_df_orig.groupby('config_id')['timestamp'].idxmax()]


        # --- Calculate latest daily return for each config_id --- 
        # Reusing helper function here
        daily_returns = portfolio_df_orig.groupby('config_id')['total_value'].apply(calculate_last_daily_return)
       

        # --- Merge data ---
        # Merge latest values with daily returns
        latest_data_merged = pd.merge(
            latest_indices,
            daily_returns.rename('daily_return'),
            on='config_id', # Merge on config_id directly
            how='left'
        )
        # Merge with config_df to get experiment names
        final_banner_data = pd.merge(
            latest_data_merged,
            config_df[['id', 'exp_name']],
            left_on='config_id',
            right_on='id',
            how='left'
        ).fillna({'exp_name': 'Unknown Exp'}) # Handle missing names

        # --- Build banner strings ---
        value_items = []
        return_items = []
        # Sort by experiment name for consistent banner order
        for _, row in final_banner_data.sort_values(by='exp_name').iterrows():
            exp_name = row['exp_name']

            # Total Value String
            latest_value = row['total_value']
            if pd.notna(latest_value):
                 value_items.append(f'<span class="exp-name">{exp_name}</span>: ${latest_value:,.2f}')
            else:
                 value_items.append(f'<span class="exp-name">{exp_name}</span>: N/A')


            # Daily Return String
            daily_return = row['daily_return']
            if pd.isna(daily_return):
                return_items.append(f'<span class="exp-name">{exp_name}</span>: N/A')
            else:
                return_class = "positive-return" if daily_return >= 0 else "negative-return"
                return_sign = "+" if daily_return >= 0 else ""
                return_items.append(f'<span class="exp-name">{exp_name}</span>: <span class="{return_class}">{return_sign}{daily_return:.2f}%</span>')

        # Join items with a separator span
        separator = '<span class="item-separator">•</span>'
        if value_items:
            latest_values_str = separator.join(value_items)
        if return_items:
            latest_returns_str = separator.join(return_items)

    except Exception as e:
        st.error(f"Error calculating banner data: {e}", icon="⚠️")
        # Ensure default strings are returned on error
        latest_values_str = "<span>Error loading banner data</span>"
        latest_returns_str = "<span>Error loading banner data</span>"

    return latest_values_str, latest_returns_str


# --- Page Display Functions ---

def display_leaderboard(config_df, portfolio_df_indexed, portfolio_df_orig):
    st.divider()
    st.title("🏆 Leaderboard for All LLM Models in Stock Market 📈")

    # --- Chart Section (Existing Logic) ---
    # Allow user to select an experiment/workflow
    exp_name_to_id = pd.Series(config_df.id.values, index=config_df.exp_name).to_dict()
    experiment_options = ["All"] + list(exp_name_to_id.keys())
    selected_exp_name = st.selectbox("Select LLM Model:", options=experiment_options, key='leaderboard_exp_select')

    # Allow user to select display type (Value or Return Period)
    display_type = st.radio(
        "Select Time Period Granularity:", # Changed label slightly
        ("Daily", "Weekly", "Monthly", "Yearly"), # Simplified options
        horizontal=True,
        key='leaderboard_display_radio'
    )

    chart_title = ""
    echarts_series = []
    legend_data = []

    # --- Set Y-Axis to Cumulative Return Percentage --- 
    y_axis_name = "Cumulative Return (%)"

    # Determine Resampling Frequency
    resample_freq = None
    if display_type == "Weekly":
        resample_freq = 'W'
    elif display_type == "Monthly":
        resample_freq = 'ME' # Month End
    elif display_type == "Yearly":
        resample_freq = 'YE' # Year End
    # 'Daily' has no resampling (resample_freq remains None)

    # --- Update Chart Title --- 
    time_period_str = display_type
    if selected_exp_name == "All":
        chart_title = f"Cumulative Return for All Experiments ({time_period_str})" 
    else:
        chart_title = f"Cumulative Return for {selected_exp_name} ({time_period_str})" 

    # --- Chart Data Preparation Loop --- 
    exp_list_to_process = []
    if selected_exp_name == "All":
        exp_list_to_process = list(exp_name_to_id.items())
    else:
        if selected_exp_name in exp_name_to_id:
             exp_list_to_process = [(selected_exp_name, exp_name_to_id[selected_exp_name])]
        else:
             st.warning(f"Experiment '{selected_exp_name}' not found in config.")
             exp_list_to_process = []

    for exp_name, config_id in exp_list_to_process:
        # Use the indexed DataFrame passed to the function
        exp_data_full = portfolio_df_indexed[portfolio_df_indexed['config_id'] == config_id].sort_index()
        if exp_data_full.empty:
            continue

        data_to_process = exp_data_full['total_value']

        # Resample if needed (Weekly, Monthly, Yearly)
        if resample_freq:
            if data_to_process.empty:
                continue
            data_to_process = data_to_process.resample(resample_freq).last()
            data_to_process = data_to_process.dropna()

        # --- Always Calculate Cumulative Return --- 
        if data_to_process.shape[0] < 2:
             continue 
        data_to_plot = calculate_cumulative_return(data_to_process)
        data_to_plot = data_to_plot.dropna() 

        if not data_to_plot.empty:
            legend_data.append(exp_name)
            # 直接使用日期作为x轴，值作为y轴
            data_pairs = [[idx.strftime('%Y-%m-%d %H:%M:%S'), round(val, 2)] for idx, val in data_to_plot.items()]
            series_name = exp_name
            echarts_series.append({
                "name": series_name,
                "data": data_pairs,
                "type": "line", 
                "smooth": True, 
                "symbol": "none",
            })

    # --- Chart Display --- 
    if echarts_series:
        option = {
            "title": {"text": chart_title, "left": "left"},
            "tooltip": {
                "trigger": 'axis',
                "axisPointer": {"type": 'cross'},
                "valueFormatter": '(value) => value ? value.toFixed(2) + "%" : "N/A"' 
            },
            "legend": {"data": legend_data, "bottom": 65, "type": 'scroll'},
            "grid": {"left": '3%', "right": '4%', "bottom": '20%', "containLabel": True},
            "xAxis": {
                "type": "time",
                "axisLabel": {"formatter": '{yyyy}-{MM}-{dd}\n{HH}:{mm}:{ss}', "rotate": 0}
            },
            "yAxis": {
                "type": "value",
                "name": y_axis_name,
                "axisLabel": {},
                "scale": True 
            },
            "series": echarts_series,
            "dataZoom": [
                {"type": "slider", "xAxisIndex": 0, "start": 0, "end": 100, "bottom": 30, "zoomLock": False},
                {"type": "inside", "xAxisIndex": 0, "start": 0, "end": 100, "zoomOnMouseWheel": False, "moveOnMouseWheel": True}
            ],
        }
        st_echarts(options=option, height="500px")
    else:
        if selected_exp_name == "All":
            st.warning(f"No suitable data found to display Cumulative Return for any experiment.")
        else:
            st.warning(f"No suitable data found to display Cumulative Return for experiment: {selected_exp_name}. Ensure sufficient data points exist.")

    # --- Summary Table / Single Experiment Dashboard Section ---
    # st.divider()

    if selected_exp_name == "All":
        # --- 添加整体市场仪表板 --- (Moved below chart)
        table_data = calculate_experiment_stats(config_df, portfolio_df_orig)

        # --- 添加所有模型与市场指数对比图表 --- (Moved to top of 'All' section)
        # st.subheader("LLMs vs Market Indices Performance")

        # 获取时间范围（使用所有模型数据的最早开始日期和最晚结束日期）
        all_timestamps = portfolio_df_orig['timestamp'].dropna()
        if not all_timestamps.empty:
            min_date = all_timestamps.min()
            max_date = all_timestamps.max()
            date_range = pd.date_range(start=min_date, end=max_date, freq='D')

            # 创建模拟的市场指数数据
            market_data = pd.DataFrame(index=date_range)

            # 模拟三大指数的初始值和变化
            np.random.seed(42)  # 确保结果一致性

            # 基本模式 + 随机波动
            nasdaq_changes = np.concatenate([np.linspace(0, 0.15, len(date_range)//2),
                                          np.linspace(0.15, 0.25, len(date_range) - len(date_range)//2)]) + np.random.normal(0, 0.02, len(date_range))
            sp500_changes = np.concatenate([np.linspace(0, 0.10, len(date_range)//2),
                                         np.linspace(0.10, 0.18, len(date_range) - len(date_range)//2)]) + np.random.normal(0, 0.015, len(date_range))
            dow_changes = np.concatenate([np.linspace(0, 0.08, len(date_range)//2),
                                       np.linspace(0.08, 0.15, len(date_range) - len(date_range)//2)]) + np.random.normal(0, 0.01, len(date_range))

            # 转换为累计收益率
            market_data['NASDAQ'] = (1 + nasdaq_changes) * 100 - 100
            market_data['S&P 500'] = (1 + sp500_changes) * 100 - 100
            market_data['DOW JONES'] = (1 + dow_changes) * 100 - 100

            # 计算所有模型的平均收益率
            # 根据配置ID分组
            grouped = portfolio_df_orig.groupby('config_id')

            # 创建一个空的DataFrame，用于存储每个模型在每个日期的累计收益率
            all_returns = pd.DataFrame(index=date_range)

            # 计算每个模型的累计收益率
            for config_id, group in grouped:
                group_sorted = group.sort_values('timestamp')

                # --- 处理重复时间戳：保留最后一条记录 ---
                group_sorted = group_sorted.drop_duplicates(subset='timestamp', keep='last')
                # -------------------------------------

                timestamps = group_sorted['timestamp']
                values = group_sorted['total_value']

                if len(values) >= 2:
                    returns = calculate_cumulative_return(values)
                    # 将收益率与时间戳对齐
                    returns_series = pd.Series(returns.values, index=timestamps)

                    # --- 确保 reindex 前索引唯一 (安全检查) ---
                    if not returns_series.index.is_unique:
                        st.warning(f"Duplicate timestamps detected for model {config_id} even after filtering. Keeping last entry.")
                        returns_series = returns_series.loc[~returns_series.index.duplicated(keep='last')] # 保留最后一个重复项
                    # ---------------------------------------

                    # 只有在索引唯一时才执行 reindex
                    if returns_series.index.is_unique:
                        # 重采样到完整日期范围
                        returns_resampled = returns_series.reindex(date_range, method='ffill')
                        # 将当前模型的收益率添加到all_returns
                        all_returns[f'Model_{config_id}'] = returns_resampled
                    else:
                        st.warning(f"Skipping model {config_id} for average calculation due to persistent duplicate timestamps.")

            # 计算所有模型的平均收益率
            if not all_returns.empty and len(all_returns.columns) > 0:
                all_returns['Average_LLMs'] = all_returns.mean(axis=1)

                # 合并LLM平均收益率和市场指数数据
                combined_data = pd.DataFrame(index=date_range)
                combined_data['Average LLMs'] = all_returns['Average_LLMs']
                combined_data['NASDAQ'] = market_data['NASDAQ']
                combined_data['S&P 500'] = market_data['S&P 500']
                combined_data['DOW JONES'] = market_data['DOW JONES']

                # 过滤掉NaN值
                combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')

                # 准备ECharts图表数据
                series = []
                for column in combined_data.columns:
                    line_type = "solid"
                    line_width = 2
                    # 为"Average LLMs"设置特殊样式
                    if column == 'Average LLMs':
                        line_width = 4
                        line_type = "solid"

                    series.append({
                        'name': column,
                        'type': 'line',
                        'data': combined_data[column].tolist(),
                        'smooth': True,
                        'symbol': 'none',
                        'lineStyle': {
                            'width': line_width,
                            'type': line_type
                        }
                    })

                # 设置ECharts图表选项
                chart_options = {
                    'title': {'text': 'LLMs vs Market Indices Comparison'},
                    'tooltip': {
                        'trigger': 'axis',
                    },
                    'legend': {
                        'data': combined_data.columns.tolist(),
                        'bottom': 45
                    },
                    'grid': {
                        'left': '3%',
                        'right': '4%',
                        'bottom': '15%',
                        'containLabel': True
                    },
                    'xAxis': {
                        'type': 'category',
                        'data': [d.strftime('%Y-%m-%d') for d in combined_data.index],
                        'axisLabel': {
                            'rotate': 45
                        }
                    },
                    'yAxis': {
                        'type': 'value',
                        'name': 'Cumulative Return (%)',
                        'axisLabel': {},
                        'scale': True
                    },
                    'series': series,
                    'dataZoom': [
                        {
                            'type': 'slider',
                            'xAxisIndex': 0,
                            'start': 0,
                            'end': 100,
                            'bottom': 10,
                            'minSpan': 20,  # 最小缩放级别 (20%)
                            'maxSpan': 100,  # 最大缩放级别 (100%)
                            'zoomLock': False
                        },
                        {
                            'type': 'inside',
                            'xAxisIndex': 0,
                            'start': 0,
                            'end': 100,
                            'minSpan': 20,
                            'maxSpan': 100,
                            'zoomOnMouseWheel': False,
                            'moveOnMouseWheel': True
                        }
                    ]
                }

                # 显示图表
                st_echarts(options=chart_options, height='500px')

                # 添加说明文本
                st.caption("Note: Market index data is simulated for demonstration purposes. In a real application, accurate historical market data should be used.")
            else:
                st.warning("Insufficient data to display comparison chart.")
        else:
            st.warning("No timestamp data available for models to create comparison chart.")
        # --- End of Moved Comparison Chart ---

        # --- Display Overall Market Dashboard --- (Now below the chart)
        if not table_data.empty:
            # 计算关键指标
            total_value = table_data['Current Total Value ($)'].sum()

            # 平均回报率
            avg_return = table_data['Total Return (%)'].mean() if 'Total Return (%)' in table_data.columns else None

            # 最佳表现模型
            best_model_idx = table_data['Total Return (%)'].idxmax() if 'Total Return (%)' in table_data.columns else None
            best_model = table_data.loc[best_model_idx, 'LLM Model'] if best_model_idx is not None else "N/A"
            best_return = table_data.loc[best_model_idx, 'Total Return (%)'] if best_model_idx is not None else None

            # 显示整体仪表板
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Investment Value", f"${total_value:,.2f}" if pd.notna(total_value) else "N/A")

            with col2:
                st.metric("Average Return", f"{avg_return:.2f}%" if pd.notna(avg_return) else "N/A")

            with col3:
                st.metric("Best Performing Model", best_model)

            with col4:
                st.metric("Best Model Return", f"{best_return:.2f}%" if pd.notna(best_return) else "N/A")
        # --- End of Metrics Section ---

        # --- Display Summary Table for All Experiments --- (Remains at the end)
        # st.subheader("Experiment Summary & Ranking")
        table_data = calculate_experiment_stats(config_df, portfolio_df_orig)

        if not table_data.empty:
            # Apply formatting before styling
            table_data_formatted = table_data.copy()
            table_data_formatted['Daily Return (%)'] = table_data_formatted['Daily Return (%)'].map(lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A')
            table_data_formatted['Total Return (%)'] = table_data_formatted['Total Return (%)'].map(lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A')
            table_data_formatted['Current Total Value ($)'] = table_data_formatted['Current Total Value ($)'].map(lambda x: f'${x:,.2f}' if pd.notna(x) else 'N/A')

            # Apply Styling
            styled_table = table_data_formatted.style.set_properties(
                subset=['Rank', 'Rank Change'],
                **{'text-align': 'left'}
            ).set_properties(
                subset=['Daily Return (%)', 'Total Return (%)', 'Current Total Value ($)'],
                **{'text-align': 'right'}
            )

            # Calculate dynamic height
            dynamic_height = (len(table_data) + 1) * 35 + 3
            st.dataframe(
                styled_table, # Use the styled table
                use_container_width=True,
                hide_index=True,
                height=dynamic_height
            )
        else:
            st.info("No experiment data available to display the summary table.")

    else: # A specific experiment is selected
        # --- Display Dashboard for the Selected Experiment ---
        # st.subheader(f"Dashboard for {selected_exp_name}")

        if selected_exp_name in exp_name_to_id:
            selected_config_id = exp_name_to_id[selected_exp_name]
            # Filter data for the selected experiment
            exp_data = portfolio_df_orig[portfolio_df_orig['config_id'] == selected_config_id].sort_values('timestamp')
            exp_config = config_df[config_df['id'] == selected_config_id].iloc[0]

            if not exp_data.empty:
                # --- Performance Trend Chart (Moved Above Metrics) ---
                # st.subheader("Performance Trend (Market Comparison)")

                # Create market benchmark data (simulated data)
                # In a real environment, this data should be obtained from financial data providers
                if not exp_data.empty:
                    # Get experiment start and end dates for creating benchmark data with the same timeframe
                    start_date_dt = exp_data['timestamp'].min() # Use datetime object
                    end_date_dt = exp_data['timestamp'].max() # Use datetime object
                    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')

                    # Create simulated market index dataframe
                    market_data = pd.DataFrame(index=date_range)

                    # Simulate initial values and changes for three major indices
                    # These are example values, should be replaced with actual historical data
                    np.random.seed(42)  # Ensure result consistency

                    # Basic pattern + random fluctuation
                    nasdaq_changes = np.concatenate([np.linspace(0, 0.15, len(date_range)//2),
                                                   np.linspace(0.15, 0.25, len(date_range) - len(date_range)//2)]) + np.random.normal(0, 0.02, len(date_range))
                    sp500_changes = np.concatenate([np.linspace(0, 0.10, len(date_range)//2),
                                                  np.linspace(0.10, 0.18, len(date_range) - len(date_range)//2)]) + np.random.normal(0, 0.015, len(date_range))
                    dow_changes = np.concatenate([np.linspace(0, 0.08, len(date_range)//2),
                                                np.linspace(0.08, 0.15, len(date_range) - len(date_range)//2)]) + np.random.normal(0, 0.01, len(date_range))

                    # Convert to cumulative returns
                    market_data['NASDAQ'] = (1 + nasdaq_changes) * 100 - 100
                    market_data['S&P 500'] = (1 + sp500_changes) * 100 - 100
                    market_data['DOW JONES'] = (1 + dow_changes) * 100 - 100

                    # Align market data dates with experiment data
                    market_data_aligned = market_data.asof(exp_data['timestamp'])
                    market_data_aligned.index = exp_data['timestamp']

                    # Calculate experiment's cumulative return
                    exp_return_series = calculate_cumulative_return(exp_data['total_value'])

                    # Merge experiment returns with market data
                    combined_data = pd.DataFrame(index=exp_data['timestamp'])
                    combined_data[f'{selected_exp_name} Return'] = exp_return_series.values
                    combined_data['NASDAQ'] = market_data_aligned['NASDAQ']
                    combined_data['S&P 500'] = market_data_aligned['S&P 500']
                    combined_data['DOW JONES'] = market_data_aligned['DOW JONES']

                    # Use st_echarts for more control over the chart
                    series = []
                    for column in combined_data.columns:
                        series.append({
                            'name': column,
                            'type': 'line',
                            'data': combined_data[column].tolist(),
                            'smooth': True,
                            'symbol': 'none'
                        })

                    chart_options = {
                        'title': {'text': 'Performance Comparison with Market Indices'},
                        'tooltip': {
                            'trigger': 'axis',
                            'valueFormatter': '(value) => value ? value.toFixed(2) + "%" : "N/A"' 
                        },
                        'legend': {
                            'data': combined_data.columns.tolist(),
                            'bottom': 45
                        },
                        'grid': {
                            'left': '3%',
                            'right': '4%',
                            'bottom': '15%',
                            'containLabel': True
                        },
                        'xAxis': {
                            'type': 'category',
                            'data': [d.strftime('%Y-%m-%d') for d in combined_data.index],
                            'axisLabel': {
                                'rotate': 45
                            }
                        },
                        'yAxis': {
                            'type': 'value',
                            'name': 'Cumulative Return (%)',
                            'axisLabel': {},
                            'scale': True
                        },
                        'series': series,
                        'dataZoom': [
                            {
                                'type': 'slider',
                                'xAxisIndex': 0,
                                'start': 0,
                                'end': 100,
                                'bottom': 10,
                                'minSpan': 20,  # Minimum zoom level (20%)
                                'maxSpan': 100,  # Maximum zoom level (100%)
                                'zoomLock': False
                            },
                            {
                                'type': 'inside',
                                'xAxisIndex': 0,
                                'start': 0,
                                'end': 100,
                                'minSpan': 20,  # Minimum zoom level (20%)
                                'maxSpan': 100,  # Maximum zoom level (100%)
                                'zoomOnMouseWheel': False,  # Disable zoom on mouse wheel
                                'moveOnMouseWheel': True   # Allow panning with mouse wheel
                            }
                        ]
                    }

                    st_echarts(options=chart_options, height='500px')

                    # Add explanation text
                    st.caption("Note: Market index data is simulated for demonstration purposes. In a real application, accurate historical market data should be used.")
                else:
                    st.warning("Unable to display performance trend: insufficient data")
                # --- End of Moved Performance Trend Chart ---

                # --- Calculate Metrics for Selected Experiment --- (Now below chart)
                latest_row = exp_data.iloc[-1]
                latest_value = latest_row['total_value']
                latest_date = latest_row['timestamp'].strftime('%Y-%m-%d')
                start_date = exp_data['timestamp'].min().strftime('%Y-%m-%d') # Already used above, keep for metrics

                daily_return = calculate_last_daily_return(exp_data['total_value'])
                cum_return_series = calculate_cumulative_return(exp_data['total_value'])
                total_return = cum_return_series.iloc[-1] if not cum_return_series.empty else None

                # --- Dashboard Layout --- 
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Return", f"{total_return:.2f}%" if pd.notna(total_return) else "N/A")
                    st.metric("Latest Daily Return", f"{daily_return:.2f}%" if pd.notna(daily_return) else "N/A")

                with col2:
                    st.metric("Current Total Value", f"${latest_value:,.2f}" if pd.notna(latest_value) else "N/A")
                    st.metric("Start Date", start_date)

                with col3:
                    # Placeholder for other info or maybe rank if we fetch it?
                    st.metric("LLM Model", exp_config.get('llm_model', 'N/A'))
                    st.metric("Update Date", latest_date)  # 添加最新更新日期
                # --- End of Metrics Section ---

                # st.divider() # Divider remains commented out

                # --- Display Holdings --- (Remains after metrics)
                st.subheader("Current Holdings")
                latest_holdings_json = latest_row['holdings']
                display_holdings_dashboard(latest_holdings_json, latest_value)

                # st.divider() # Divider remains commented out

                # --- Optional: Mini Chart --- (This section is now empty as the chart moved)

            else:
                st.warning(f"No portfolio data found for experiment: {selected_exp_name}")
        else:
            # This case should technically not be reached due to earlier check
            st.error(f"Selected experiment '{selected_exp_name}' not found.")

# --- Function to Display Holdings in Dashboard ---
def display_holdings_dashboard(holdings_json_str, total_value):
    """Parses holdings JSON and displays them in a structured format for the dashboard."""
    if not holdings_json_str or holdings_json_str == '{}' or pd.isna(holdings_json_str):
        st.info("No current holdings data available.")
        return

    try:
        holdings_dict = json.loads(holdings_json_str)
        if not holdings_dict:
            st.info("Portfolio currently holds no assets (all cash).")
            return

        holdings_list = []
        for ticker, data in holdings_dict.items():
            value = data.get('value', 0)
            shares = data.get('shares', 'N/A')
            percentage = (value / total_value) * 100 if total_value and pd.notna(total_value) and total_value != 0 else 0
            holdings_list.append({
                'Ticker': ticker,
                'Shares': shares,
                'Current Value ($)': value,
                '% of Portfolio': percentage
            })

        if not holdings_list:
            st.info("Portfolio currently holds no assets (all cash).")
            return

        holdings_df = pd.DataFrame(holdings_list)
        holdings_df = holdings_df.sort_values(by='% of Portfolio', ascending=False)

        # Format for display
        holdings_df_formatted = holdings_df.copy()
        holdings_df_formatted['Current Value ($)'] = holdings_df_formatted['Current Value ($)'].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        holdings_df_formatted['% of Portfolio'] = holdings_df_formatted['% of Portfolio'].map(lambda x: f"{x:.2f}%")

        # Use columns for better layout potentially
        # st.dataframe(holdings_df_formatted, hide_index=True, use_container_width=True)

        # Alternative: Display using st.metric or similar in columns for a card-like view
        num_holdings = len(holdings_df)
        cols_per_row = 4 # Adjust as needed
        num_rows = (num_holdings + cols_per_row - 1) // cols_per_row

        idx = 0
        for r in range(num_rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                if idx < num_holdings:
                    holding = holdings_df.iloc[idx]
                    with cols[c]:
                        st.metric(
                            label=f"{holding['Ticker']}", 
                            value=f"${holding['Current Value ($)']:,.2f}",
                            delta=f"{holding['% of Portfolio']:.1f}% | {holding['Shares'] if holding['Shares'] != 'N/A' else '-'} Shares"
                        )
                    idx += 1
                # else: # Optional: Add empty containers to fill row
                #     with cols[c]:
                #         st.empty()

    except json.JSONDecodeError:
        st.error("Failed to parse holdings data.")
    except Exception as e:
        st.error(f"An error occurred displaying holdings: {e}")
        st.exception(e)

def show_agent_lab():
    # 调用导入的函数
    display_agent_lab()

def show_markets():
    # 调用导入的函数
    display_markets() 

def show_reports():
    # 调用导入的函数
    display_reports() 

def show_about_us():
    # 调用导入的函数
    display_about_us() 

def show_community():
    # 调用导入的函数
    display_community()

# --- Main App Logic --- 

# Initialize data variables
config_df = None
portfolio_df_orig = None
portfolio_df_indexed = None
data_loaded_successfully = False

# Load data first
try:
    config_df = pd.read_csv('data/config_rows.csv')
    if 'id' not in config_df.columns or 'exp_name' not in config_df.columns:
         raise ValueError("Config file must contain 'id' and 'exp_name' columns.")

    portfolio_cols = ['portfolio_id', 'config_id', 'timestamp', 'cash', 'total_value', 'holdings']
    portfolio_df_orig = pd.read_csv('data/portfolio_rows.csv', names=portfolio_cols, header=None, skiprows=1)

    if not portfolio_df_orig.empty:
        if not all(col in portfolio_df_orig.columns for col in ['config_id', 'timestamp', 'total_value', 'holdings']): # Added holdings check
             raise ValueError("Portfolio file must contain 'config_id', 'timestamp', 'total_value', 'holdings' columns.")
        portfolio_df_orig['timestamp'] = pd.to_datetime(portfolio_df_orig['timestamp'], errors='coerce')
        portfolio_df_orig.dropna(subset=['timestamp', 'config_id', 'total_value'], inplace=True) # Keep rows even if holdings are NaN initially
        portfolio_df_indexed = portfolio_df_orig.set_index('timestamp').copy()
    else:
        # Define columns for empty DataFrames to avoid errors later
        portfolio_df_indexed = pd.DataFrame(columns=portfolio_cols[1:])
        portfolio_df_indexed['timestamp'] = pd.to_datetime([])
        portfolio_df_indexed = portfolio_df_indexed.set_index('timestamp')
        portfolio_df_orig = pd.DataFrame(columns=portfolio_cols)


    data_loaded_successfully = True

except FileNotFoundError as e:
    st.error(f"Error loading data file: {e}. Please ensure 'data/config_rows.csv' and 'data/portfolio_rows.csv' exist.")
except pd.errors.EmptyDataError as e:
    st.warning(f"Data file is empty: {e}. Some features might be unavailable.")
    if config_df is not None and portfolio_df_orig is None:
         portfolio_df_orig = pd.DataFrame(columns=portfolio_cols)
         portfolio_df_indexed = portfolio_df_orig.set_index('timestamp') # Set index on empty df
         data_loaded_successfully = True
    elif config_df is None:
         data_loaded_successfully = False
except ValueError as e:
     st.error(f"Data validation error: {e}")
except Exception as e:
    st.error(f"An critical error occurred during initial data loading: {e}")
    st.exception(e)

# --- Banner Calculation and Display (Runs regardless of page) ---
# TODO: Refactor banner calculation potentially using parts of calculate_experiment_stats
top_banner_content, bottom_banner_content = get_banner_data(config_df, portfolio_df_orig)

# Define separator for duplication
separator = '<span class="item-separator">•</span>'
# Duplicate content for seamless looping (if content exists)
duplicated_top_content = f"{top_banner_content}{separator}{top_banner_content}" if top_banner_content and "No portfolio data" not in top_banner_content else top_banner_content
duplicated_bottom_content = f"{bottom_banner_content}{separator}{bottom_banner_content}" if bottom_banner_content and "No portfolio data" not in bottom_banner_content else bottom_banner_content

st.markdown(marquee_css, unsafe_allow_html=True)
st.markdown(f'<div class="marquee-container top-banner"><div class="marquee-content">{duplicated_top_content}</div></div>', unsafe_allow_html=True)
st.markdown(f'<div class="marquee-container bottom-banner"><div class="marquee-content">{duplicated_bottom_content}</div></div>', unsafe_allow_html=True)

# --- Page Title and Navigation ---
st.title("💰DEEPFUND🔥--The First AI Live Investment Arena")

# --- Links and Introduction Text ---
links_md = """ 
<span style="font-size: 1.3em;">

[WeChat](#WeChat) | [Twitter](#twitter) | [小红书](https://www.xiaohongshu.com/user/profile/67f78832000000000e011700) | [GitHub](https://github.com/HKUSTDial/deepfund) | [Paper](http://arxiv.org/abs/2503.18313) | [BiliBili](#BiliBili)
"""
st.markdown(links_md, unsafe_allow_html=True)

intro_md = """
<span style="font-size: 1.3em;">

**Will LLMs Be Professional At Fund Investment?**

We evaluate the trading capability of LLM across various financial markets given a unified environment. 
The LLM shall ingest external information, drive a multi-agent system, and make trading decisions. 
The LLM performance will be presented in a trading arena view across various dimensions.
<br><br>
**DeepFund Arena thrives on community engagement — you can customize your own LLM agent and compete with others!**

<span style="display: inline-block; border: 2px solid #17a2b8; border-radius: 8px; padding: 8px 15px; color: #17a2b8;">
Join us to explore the future of AI investment by <a href="https://github.com/HKUSTDial/deepfund" target="_blank" style="color: #17a2b8; text-decoration: none;">⚙️ Customize Your Own LLM Agent</a>
</span>
"""
st.markdown(intro_md, unsafe_allow_html=True)


# st.divider() # Add a visual separator before the menu (Commented out)

# --- Main Content Area (Conditional on Data Load) ---
if data_loaded_successfully and config_df is not None and portfolio_df_indexed is not None and portfolio_df_orig is not None:
    # --- Move Menu to Sidebar --- 
    with st.sidebar:
        # --- Add Logo --- 
        st.image("image/logodeepfund.png") # Display the logo image
        # --- End Logo ---
        
        selected_page = option_menu(
            menu_title=None, # Remove text title, logo is now the title
            options=["Leaderboard", "Agent Lab", "Community", "Markets", "Reports", "About Us"],
            icons=['graph-up','robot', 'people', 'coin', 'newspaper',  'info-circle'],
            menu_icon="wallet2", 
            default_index=0,
            orientation="vertical", 
            styles={ 
                "container": {"padding": "5px !important", "background-color": "#fafafa"}, 
                "icon": {"color": "orange", "font-size": "23px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "padding": "10px", "--hover-color": "#eee"}, 
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
    # --- End Sidebar Menu ---

    # --- Page Display Logic (remains outside sidebar) ---
    # Display the selected page content
    if selected_page == "Leaderboard":
        # Pass portfolio_df_orig to the function as well
        display_leaderboard(config_df, portfolio_df_indexed, portfolio_df_orig)
    elif selected_page == "Agent Lab":
        show_agent_lab()
    elif selected_page == "Markets":
        show_markets() 
    elif selected_page == "Reports":
        show_reports() 
    elif selected_page == "Community":
        show_community()
    elif selected_page == "About Us":
        show_about_us()

else:
     st.warning("App cannot display main content because data failed to load correctly.")