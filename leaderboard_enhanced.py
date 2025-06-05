import streamlit as st
import pandas as pd
import numpy as np
import json
from streamlit_echarts import st_echarts
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def load_enhanced_data():
    """加载新的数据结构并进行预处理"""
    try:
        # 加载配置数据
        config_df = pd.read_csv('data/config_rows_new.csv')
        
        # 加载投资组合数据  
        portfolio_df = pd.read_csv('data/portfolio_rows_new.csv')
        
        # 重命名字段以保持兼容性
        portfolio_df = portfolio_df.rename(columns={
            'updated_at': 'timestamp',
            'total_assets': 'total_value',
            'positions': 'holdings'
        })
        
        # 时间字段处理
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df['trading_date'] = pd.to_datetime(portfolio_df['trading_date'])
        
        # 数据清理
        portfolio_df = portfolio_df.dropna(subset=['timestamp', 'config_id', 'total_value'])
        portfolio_df = portfolio_df.sort_values(['config_id', 'timestamp'])
        portfolio_df = portfolio_df.drop_duplicates(subset=['config_id', 'timestamp'], keep='last')
        
        return config_df, portfolio_df
        
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

def calculate_advanced_metrics(portfolio_df, config_df):
    """计算高级性能指标"""
    if portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame()
    
    results = []
    
    for config_id in portfolio_df['config_id'].unique():
        agent_data = portfolio_df[portfolio_df['config_id'] == config_id].sort_values('timestamp')
        config_info = config_df[config_df['id'] == config_id].iloc[0] if config_id in config_df['id'].values else None
        
        if len(agent_data) < 2:
            continue
            
        # 基础信息
        exp_name = config_info['exp_name'] if config_info is not None else f"Agent-{config_id[:8]}"
        llm_model = config_info['llm_model'] if config_info is not None else "Unknown"
        
        # 计算收益率序列
        values = agent_data['total_value'].values
        returns = np.diff(values) / values[:-1]
        
        # 累计收益率
        total_return = (values[-1] / values[0] - 1) * 100
        
        # 年化收益率（假设数据跨度）
        days = (agent_data['timestamp'].max() - agent_data['timestamp'].min()).days
        annual_return = ((values[-1] / values[0]) ** (365/max(days, 1)) - 1) * 100 if days > 0 else 0
        
        # 波动率（年化）
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0
        
        # 夏普比率（假设无风险利率为2%）
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate * 100) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # 胜率
        win_rate = np.sum(returns > 0) / len(returns) * 100 if len(returns) > 0 else 0
        
        # 当前持仓分析
        latest_holdings = agent_data.iloc[-1]['holdings']
        num_positions = 0
        try:
            if latest_holdings and latest_holdings != '{}':
                holdings_dict = json.loads(latest_holdings)
                num_positions = len(holdings_dict)
        except:
            pass
        
        # 格式化持仓信息
        def format_holdings(holdings_json_str, total_value):
            """解析并格式化持仓信息"""
            if not holdings_json_str or holdings_json_str == '{}' or pd.isna(holdings_json_str) or total_value == 0 or pd.isna(total_value):
                return "N/A"
            try:
                holdings_dict = json.loads(holdings_json_str)
                if not holdings_dict:
                    return "Cash 100%"
                
                items = []
                for ticker, data in holdings_dict.items():
                    value = data.get('value', 0)
                    percentage = (value / total_value) * 100 if total_value else 0
                    items.append(f"{ticker}: {percentage:.1f}%")
                return ", ".join(items)
            except json.JSONDecodeError:
                return "Invalid Data"
            except Exception:
                return "Error"
        
        # 获取分析师投资组合信息
        latest_total_value = values[-1]
        analyst_portfolio = format_holdings(latest_holdings, latest_total_value)
        
        results.append({
            'config_id': config_id,
            'Agent Name': exp_name,
            'LLM Model': llm_model,
            'Start Date': agent_data['trading_date'].min().strftime('%Y-%m-%d'),
            'Total Return (%)': total_return,
            'Annual Return (%)': annual_return,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Current Value ($)': values[-1],
            'Analyst Portfolio': analyst_portfolio,
            'Positions': num_positions,
            'Trading Days': days,
            'End Date': agent_data['trading_date'].max().strftime('%Y-%m-%d')
        })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Total Return (%)', ascending=False)
        results_df['Rank'] = range(1, len(results_df) + 1)
    
    return results_df

def create_performance_comparison_chart(portfolio_df, config_df):
    """创建性能对比图表"""
    fig = go.Figure()
    
    for config_id in portfolio_df['config_id'].unique():
        agent_data = portfolio_df[portfolio_df['config_id'] == config_id].sort_values('timestamp')
        config_info = config_df[config_df['id'] == config_id].iloc[0] if config_id in config_df['id'].values else None
        exp_name = config_info['exp_name'] if config_info is not None else f"Agent-{config_id[:8]}"
        
        if len(agent_data) >= 2:
            values = agent_data['total_value'].values
            cumulative_returns = (values / values[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=agent_data['trading_date'],
                y=cumulative_returns,
                mode='lines',
                name=exp_name,
                line=dict(width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Return: %{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title='LLM Agents Performance Comparison',
        xaxis_title='Trading Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    return fig

def create_individual_agent_analysis(portfolio_df, config_df, selected_agent_id):
    """创建单个代理的详细分析图表"""
    agent_data = portfolio_df[portfolio_df['config_id'] == selected_agent_id].sort_values('timestamp')
    config_info = config_df[config_df['id'] == selected_agent_id].iloc[0] if selected_agent_id in config_df['id'].values else None
    agent_name = config_info['exp_name'] if config_info is not None else f"Agent-{selected_agent_id[:8]}"
    
    if len(agent_data) < 2:
        return None
    
    # 计算各种指标
    values = agent_data['total_value'].values
    cumulative_returns = (values / values[0] - 1) * 100
    daily_returns = np.diff(values) / values[:-1] * 100
    
    # 创建子图
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Cumulative Return Over Time', 'Daily Returns Distribution',
            'Portfolio Value Evolution', 'Running Max Drawdown',
            'Daily Returns Timeline', 'Holdings Distribution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # 1. 累计收益率走势
    fig.add_trace(
        go.Scatter(
            x=agent_data['trading_date'],
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Return',
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    
    # 2. 日收益率分布直方图
    fig.add_trace(
        go.Histogram(
            x=daily_returns,
            nbinsx=20,
            name='Daily Returns',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # 3. 投资组合价值演化
    fig.add_trace(
        go.Scatter(
            x=agent_data['trading_date'],
            y=values,
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # 4. 回撤分析
    cumulative = np.cumprod(1 + daily_returns / 100)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    
    fig.add_trace(
        go.Scatter(
            x=agent_data['trading_date'][1:],  # 去掉第一个日期因为没有收益率
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tonexty'
        ),
        row=2, col=2
    )
    
    # 5. 日收益率时间线
    fig.add_trace(
        go.Scatter(
            x=agent_data['trading_date'][1:],
            y=daily_returns,
            mode='markers+lines',
            name='Daily Returns',
            line=dict(color='orange', width=1),
            marker=dict(
                size=6,
                color=daily_returns,
                colorscale='RdYlGn',
                showscale=False
            )
        ),
        row=3, col=1
    )
    
    # 6. 最新持仓分布饼图
    latest_holdings = agent_data.iloc[-1]['holdings']
    try:
        if latest_holdings and latest_holdings != '{}':
            holdings_dict = json.loads(latest_holdings)
            if holdings_dict:
                tickers = list(holdings_dict.keys())
                values_holdings = [holdings_dict[ticker].get('value', 0) for ticker in tickers]
                
                fig.add_trace(
                    go.Pie(
                        labels=tickers,
                        values=values_holdings,
                        name="Holdings",
                        textinfo='label+percent'
                    ),
                    row=3, col=2
                )
    except:
        # 如果解析失败，显示空饼图
        fig.add_trace(
            go.Pie(
                labels=["No Data"],
                values=[1],
                name="Holdings"
            ),
            row=3, col=2
        )
    
    fig.update_layout(
        height=900,
        title_text=f"Detailed Analysis for {agent_name}",
        showlegend=True
    )
    
    return fig

def create_performance_radar_chart(metrics_df):
    """创建性能雷达图"""
    if metrics_df.empty:
        return None
    
    # 选择前5名进行雷达图比较
    top_agents = metrics_df.head(5)
    
    # 标准化指标（0-100分制）
    categories = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Low Volatility', 'Low Drawdown']
    
    fig = go.Figure()
    
    for _, agent in top_agents.iterrows():
        # 计算标准化分数
        total_return_score = min(100, max(0, (agent['Total Return (%)'] + 50) * 2))  # -50% to 50% -> 0 to 100
        sharpe_score = min(100, max(0, (agent['Sharpe Ratio'] + 2) * 25))  # -2 to 2 -> 0 to 100
        win_rate_score = agent['Win Rate (%)']  # 已经是百分比
        volatility_score = max(0, 100 - agent['Volatility (%)'] * 2)  # 低波动率得高分
        drawdown_score = max(0, 100 + agent['Max Drawdown (%)'] * 2)  # 低回撤得高分
        
        fig.add_trace(go.Scatterpolar(
            r=[total_return_score, sharpe_score, win_rate_score, volatility_score, drawdown_score],
            theta=categories,
            fill='toself',
            name=agent['Agent Name'],
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Top 5 Agents Performance Radar Chart",
        height=500
    )
    
    return fig

def create_risk_metrics_heatmap(metrics_df):
    """创建风险指标热力图"""
    if metrics_df.empty:
        return None
    
    # 选择要显示的指标
    risk_metrics = ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
    
    # 创建热力图数据
    heatmap_data = metrics_df[['Agent Name'] + risk_metrics].set_index('Agent Name')
    
    # 标准化数据用于颜色映射
    heatmap_normalized = heatmap_data.copy()
    for col in risk_metrics:
        if col in ['Volatility (%)', 'Max Drawdown (%)']:
            # 对于风险指标，值越小越好（颜色越绿）
            heatmap_normalized[col] = -heatmap_data[col]
        else:
            # 对于收益指标，值越大越好
            heatmap_normalized[col] = heatmap_data[col]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_normalized.values,
        x=risk_metrics,
        y=heatmap_normalized.index,
        colorscale='RdYlGn',
        text=heatmap_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Risk-Return Heatmap (Green=Better, Red=Worse)",
        xaxis_title="Metrics",
        yaxis_title="Agents",
        height=400
    )
    
    return fig

def create_performance_ranking_chart(metrics_df):
    """创建性能排名对比图"""
    if metrics_df.empty:
        return None
    
    # 创建排名对比的水平条形图
    fig = go.Figure()
    
    # 按总收益率排序
    sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
    
    # 添加总收益率条形图
    fig.add_trace(go.Bar(
        y=sorted_df['Agent Name'],
        x=sorted_df['Total Return (%)'],
        name='Total Return (%)',
        orientation='h',
        marker=dict(
            color=sorted_df['Total Return (%)'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Return %")
        ),
        text=[f"{val:.1f}%" for val in sorted_df['Total Return (%)']],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Agent Performance Ranking by Total Return",
        xaxis_title="Total Return (%)",
        yaxis_title="Agents",
        height=max(400, len(sorted_df) * 30),
        showlegend=False
    )
    
    return fig

def create_holdings_distribution_chart(portfolio_df):
    """创建持仓分布图表"""
    latest_data = portfolio_df.groupby('config_id').last().reset_index()
    
    all_tickers = {}
    for _, row in latest_data.iterrows():
        try:
            if row['holdings'] and row['holdings'] != '{}':
                holdings = json.loads(row['holdings'])
                for ticker, data in holdings.items():
                    value = data.get('value', 0)
                    if ticker not in all_tickers:
                        all_tickers[ticker] = 0
                    all_tickers[ticker] += value
        except:
            continue
    
    if not all_tickers:
        return None
        
    tickers_df = pd.DataFrame(list(all_tickers.items()), columns=['Ticker', 'Total Value'])
    tickers_df = tickers_df.sort_values('Total Value', ascending=True)
    
    fig = px.bar(
        tickers_df.tail(10),  # 只显示前10个
        x='Total Value',
        y='Ticker',
        orientation='h',
        title='Top Holdings Across All Agents',
        text='Total Value'
    )
    
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(height=400)
    
    return fig

def display_enhanced_leaderboard():
    """显示增强版排行榜"""
    st.title("🚀 Enhanced LLM Trading Leaderboard")
    
    # 加载数据
    config_df, portfolio_df = load_enhanced_data()
    
    if config_df is None or portfolio_df is None:
        st.error("无法加载数据，请检查数据文件")
        return
    
    # 计算高级指标
    metrics_df = calculate_advanced_metrics(portfolio_df, config_df)
    
    if metrics_df.empty:
        st.warning("没有足够的数据进行分析")
        return
    
    # 总体统计
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Agents", len(metrics_df))
    with col2:
        avg_return = metrics_df['Total Return (%)'].mean()
        st.metric("Average Return", f"{avg_return:.2f}%")
    with col3:
        best_agent = metrics_df.iloc[0]['Agent Name']
        st.metric("Top Performer", best_agent)
    with col4:
        total_value = metrics_df['Current Value ($)'].sum()
        st.metric("Total AUM", f"${total_value:,.0f}")
    
    # 性能对比图
    st.subheader("📈 Performance Comparison")
    perf_chart = create_performance_comparison_chart(portfolio_df, config_df)
    st.plotly_chart(perf_chart, use_container_width=True)
    
    # 详细排行榜
    st.subheader("🏆 Detailed Rankings")
    
    # 格式化数据用于显示
    display_df = metrics_df.copy()
    display_df['Total Return (%)'] = display_df['Total Return (%)'].map(lambda x: f"{x:.2f}%")
    display_df['Annual Return (%)'] = display_df['Annual Return (%)'].map(lambda x: f"{x:.2f}%")
    display_df['Volatility (%)'] = display_df['Volatility (%)'].map(lambda x: f"{x:.2f}%")
    display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].map(lambda x: f"{x:.3f}")
    display_df['Max Drawdown (%)'] = display_df['Max Drawdown (%)'].map(lambda x: f"{x:.2f}%")
    display_df['Win Rate (%)'] = display_df['Win Rate (%)'].map(lambda x: f"{x:.1f}%")
    display_df['Current Value ($)'] = display_df['Current Value ($)'].map(lambda x: f"${x:,.2f}")
    
    # 选择要显示的列
    columns_to_show = [
        'Rank', 'Agent Name', 'LLM Model', 'Start Date', 'Total Return (%)', 
        'Annual Return (%)', 'Volatility (%)', 'Sharpe Ratio', 
        'Max Drawdown (%)', 'Win Rate (%)', 'Current Value ($)', 
        'Analyst Portfolio', 'Positions', 'Trading Days'
    ]
    
    st.dataframe(
        display_df[columns_to_show],
        use_container_width=True,
        hide_index=True
    )
    
    # 新增：单个代理分析部分
    st.subheader("🔍 Individual Agent Analysis")
    
    # 创建代理选择下拉菜单
    agent_options = {}
    for _, row in metrics_df.iterrows():
        agent_options[row['Agent Name']] = row['config_id']
    
    selected_agent_name = st.selectbox(
        "Select an agent for detailed analysis:",
        options=list(agent_options.keys()),
        key='agent_selector'
    )
    
    if selected_agent_name:
        selected_agent_id = agent_options[selected_agent_name]
        individual_chart = create_individual_agent_analysis(portfolio_df, config_df, selected_agent_id)
        
        if individual_chart:
            st.plotly_chart(individual_chart, use_container_width=True)
        else:
            st.warning(f"Insufficient data for {selected_agent_name}")
    
    # 替换原有的risk-return分析为更直观的可视化
    st.subheader("📊 Performance Analytics Dashboard")
    
    # 创建三列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Performance Radar Chart")
        radar_chart = create_performance_radar_chart(metrics_df)
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Performance Ranking")
        ranking_chart = create_performance_ranking_chart(metrics_df)
        if ranking_chart:
            st.plotly_chart(ranking_chart, use_container_width=True)
    
    # 风险指标热力图
    st.subheader("🌡️ Risk-Return Heatmap")
    heatmap_chart = create_risk_metrics_heatmap(metrics_df)
    if heatmap_chart:
        st.plotly_chart(heatmap_chart, use_container_width=True)
    
    # 持仓分布（保留原有功能）
    st.subheader("💼 Holdings Distribution")
    holdings_chart = create_holdings_distribution_chart(portfolio_df)
    if holdings_chart:
        st.plotly_chart(holdings_chart, use_container_width=True)

if __name__ == "__main__":
    display_enhanced_leaderboard() 