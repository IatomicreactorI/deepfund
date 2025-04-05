import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
from streamlit_option_menu import option_menu
import json # Needed for parsing holdings

st.set_page_config(layout="wide")

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
    padding-top: 60px !important;  /* Adjust if top banner height changes */
    padding-bottom: 50px !important; /* Adjust if bottom banner height changes */
}
/* Fallback/Alternative targets */
div[data-testid="stAppViewContainer"] > section > div[data-testid="stVerticalBlock"] {
    padding-top: 60px !important;
    padding-bottom: 50px !important;
}
div.block-container {
    padding-top: 60px !important;
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
    """Calculates detailed stats for each experiment for the leaderboard table."""
    if portfolio_df_orig is None or portfolio_df_orig.empty or config_df is None or config_df.empty:
        return pd.DataFrame() # Return empty dataframe if no data

    all_stats = []

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(portfolio_df_orig['timestamp']):
        portfolio_df_orig['timestamp'] = pd.to_datetime(portfolio_df_orig['timestamp'], errors='coerce')
        portfolio_df_orig.dropna(subset=['timestamp'], inplace=True)

    grouped = portfolio_df_orig.groupby('config_id')

    for config_id, group in grouped:
        if group.empty:
            continue

        group_sorted = group.sort_values('timestamp')
        
        # Basic Info
        start_date = group_sorted['timestamp'].min().strftime('%Y-%m-%d')
        latest_row = group_sorted.iloc[-1]
        latest_total_value = latest_row['total_value']
        latest_holdings_str = latest_row['holdings']

        # Calculations
        daily_return_pct = calculate_last_daily_return(group_sorted['total_value'])
        total_cumulative_return_series = calculate_cumulative_return(group_sorted['total_value'])
        total_return_pct = total_cumulative_return_series.iloc[-1] if not total_cumulative_return_series.empty else None

        # Format Holdings
        holdings_formatted = format_holdings(latest_holdings_str, latest_total_value)

        # Get Config Info
        config_info = config_df[config_df['id'] == config_id].iloc[0] if config_id in config_df['id'].values else None
        exp_name = config_info['exp_name'] if config_info is not None else f"Unknown ({config_id[:6]}...)"
        llm_model = config_info['llm_model'] if config_info is not None else "N/A"

        all_stats.append({
            'config_id': config_id,
            'LLM Model': llm_model,
            'Start Date': start_date,
            'Decision Accuracy (%)': 'N/A', # Placeholder
            'Daily Return (%)': daily_return_pct,
            'Analyst Portfolio': 'N/A', # Placeholder
            'Total Return (%)': total_return_pct,
            'Current Total Value ($)': latest_total_value,
            'Current Holdings (%)': holdings_formatted,
            'Composite Score': 'N/A', # Placeholder
            'License': 'N/A', # Placeholder
            'API Cost ($)': 'N/A' # Placeholder
        })

    if not all_stats:
        return pd.DataFrame()

    stats_df = pd.DataFrame(all_stats)

    # Calculate Rank based on Total Return (%)
    stats_df = stats_df.sort_values(by='Total Return (%)', ascending=False, na_position='last')
    stats_df.insert(0, 'Rank', range(1, len(stats_df) + 1))

    # Format numeric columns
    stats_df['Daily Return (%)'] = stats_df['Daily Return (%)'].map(lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A')
    stats_df['Total Return (%)'] = stats_df['Total Return (%)'].map(lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A')
    stats_df['Current Total Value ($)'] = stats_df['Current Total Value ($)'].map(lambda x: f'${x:,.2f}' if pd.notna(x) else 'N/A')

    # Select and reorder columns for final display
    final_columns = [
        'Rank',
        'LLM Model',
        'Start Date',
        'Decision Accuracy (%)',
        'Daily Return (%)',
        'Analyst Portfolio',
        'Total Return (%)',
        'Current Total Value ($)',
        'Current Holdings (%)',
        'Composite Score',
        'License',
        'API Cost ($)'
    ]
    # Ensure only existing columns are selected
    final_columns = [col for col in final_columns if col in stats_df.columns] 
    return stats_df[final_columns]

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
    st.subheader("Overall Leaderboard for All LLM Models in American Stock Market")

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
    y_axis_formatter = '{value}%'
    tooltip_value_formatter = "function (value) { if (value == null) return 'N/A'; return value.toFixed(2) + '%'; }"

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
            # Convert index to string for ECharts AND round the value
            data_pairs = [[idx.strftime('%Y-%m-%d %H:%M:%S'), round(val, 2)] for idx, val in data_to_plot.items()]
            series_name = exp_name
            echarts_series.append({
                "name": series_name,
                "data": data_pairs,
                "type": "line", "smooth": True, "symbol": "none",
            })

    # --- Chart Display --- 
    if echarts_series:
        option = {
            "title": {"text": chart_title, "left": "center"},
            "tooltip": {
                "trigger": 'axis',
                "axisPointer": {"type": 'cross'},
                "valueFormatter": tooltip_value_formatter # Use the % formatter
            },
            "legend": {"data": legend_data, "top": 'bottom', "type": 'scroll'},
            "grid": {"left": '3%', "right": '4%', "bottom": '15%', "containLabel": True},
            "xAxis": {
                "type": "time",
                "axisLabel": {"formatter": '{yyyy}-{MM}-{dd}\n{HH}:{mm}:{ss}', "rotate": 0}
            },
            "yAxis": {
                "type": "value",
                "name": y_axis_name, # Use "Cumulative Return (%)"
                "axisLabel": {"formatter": y_axis_formatter}, # Use "{value}%"
                "scale": True 
            },
            "series": echarts_series,
            "dataZoom": [
                {"type": "slider", "xAxisIndex": 0, "start": 0, "end": 100, "bottom": 30},
                {"type": "inside", "xAxisIndex": 0, "start": 0, "end": 100}
            ],
        }
        st_echarts(options=option, height="500px")
    else:
        if selected_exp_name == "All":
            st.warning(f"No suitable data found to display Cumulative Return for any experiment.")
        else:
            st.warning(f"No suitable data found to display Cumulative Return for experiment: {selected_exp_name}. Ensure sufficient data points exist.")

    # --- Summary Table Section --- 
    st.divider() # Add a visual separator
    st.subheader("Experiment Summary & Ranking")

    # Calculate table data using the new helper function
    table_data = calculate_experiment_stats(config_df, portfolio_df_orig)

    if not table_data.empty:
        # Calculate dynamic height: (num_rows + header) * pixels_per_row + buffer
        dynamic_height = (len(table_data) + 1) * 35 + 3 
        st.dataframe(
            table_data, 
            use_container_width=True, 
            hide_index=True, 
            height=dynamic_height # Set the calculated height
        )
    else:
        st.info("No experiment data available to display the summary table.")

def display_agent_lab():
    st.subheader("Agent Lab")
    st.write("Content for Agent Lab page goes here.")

def display_about_us():
    st.subheader("About Us")
    st.write("Content for About Us page goes here.")

def display_markets():
    st.subheader("Markets")
    st.write("Content for Markets page goes here.")

def display_reports():
    st.subheader("Reports")
    st.write("Content for Reports page goes here.")

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
[WeChat](#WeChat) | [Twitter](#twitter) | [小红书](#小红书) | [GitHub](#github) | [Paper](#paper) | [BiliBili](#BiliBili)
"""
st.markdown(links_md, unsafe_allow_html=True)

intro_md = """
<span style="font-size: 1.3em;">

**Will LLMs Be Professional At Fund Investment?**

We evaluate the trading capability of LLM across various financial markets given a unified environment. 
The LLM shall ingest external information, drive a multi-agent system, and make trading decisions. 
The LLM performance will be presented in a trading arena view across various dimensions.
<br><br>
**DeepFund Arena thrives on community engagement — cast your vote to help improve AI evaluation!**
</span>
"""
st.markdown(intro_md, unsafe_allow_html=True)


st.divider() # Add a visual separator before the menu

# --- Main Content Area (Conditional on Data Load) ---
if data_loaded_successfully and config_df is not None and portfolio_df_indexed is not None and portfolio_df_orig is not None:
    selected_page = option_menu(
        menu_title=None,
        options=["Leaderboard", "Agent Lab", "About Us", "Markets", "Reports"],
        icons=['graph-up', 'robot', 'info-circle', 'bank', 'file-earmark-text'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa", "margin-bottom": "15px"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )

    # Display the selected page content
    if selected_page == "Leaderboard":
        # Pass portfolio_df_orig to the function as well
        display_leaderboard(config_df, portfolio_df_indexed, portfolio_df_orig)
    elif selected_page == "Agent Lab":
        display_agent_lab()
    elif selected_page == "About Us":
        display_about_us()
    elif selected_page == "Markets":
        display_markets()
    elif selected_page == "Reports":
        display_reports()

else:
     st.warning("App cannot display main content because data failed to load correctly.")