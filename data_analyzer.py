import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class DataAnalyzer:
    """数据分析器类，专门处理新数据结构的分析功能"""
    
    def __init__(self, config_df: pd.DataFrame, portfolio_df: pd.DataFrame):
        """
        初始化数据分析器
        
        Args:
            config_df: 配置数据DataFrame
            portfolio_df: 投资组合数据DataFrame
        """
        self.config_df = config_df
        self.portfolio_df = portfolio_df
        self._prepare_data()
    
    def _prepare_data(self):
        """数据预处理"""
        # 确保时间字段为datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.portfolio_df['timestamp']):
            self.portfolio_df['timestamp'] = pd.to_datetime(self.portfolio_df['timestamp'])
        
        if 'trading_date' in self.portfolio_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.portfolio_df['trading_date']):
                self.portfolio_df['trading_date'] = pd.to_datetime(self.portfolio_df['trading_date'])
        
        # 数据清理和排序
        self.portfolio_df = self.portfolio_df.dropna(subset=['timestamp', 'config_id', 'total_value'])
        self.portfolio_df = self.portfolio_df.sort_values(['config_id', 'timestamp'])
        self.portfolio_df = self.portfolio_df.drop_duplicates(subset=['config_id', 'timestamp'], keep='last')
    
    def get_agent_performance(self, config_id: str) -> Dict:
        """
        获取特定代理的性能指标
        
        Args:
            config_id: 配置ID
            
        Returns:
            包含性能指标的字典
        """
        agent_data = self.portfolio_df[self.portfolio_df['config_id'] == config_id].sort_values('timestamp')
        
        if len(agent_data) < 2:
            return {}
        
        # 基础信息
        config_info = self.config_df[self.config_df['id'] == config_id].iloc[0] if config_id in self.config_df['id'].values else None
        exp_name = config_info['exp_name'] if config_info is not None else f"Agent-{config_id[:8]}"
        llm_model = config_info['llm_model'] if config_info is not None else "Unknown"
        
        # 计算性能指标
        values = agent_data['total_value'].values
        returns = self._calculate_returns(values)
        
        return {
            'agent_name': exp_name,
            'llm_model': llm_model,
            'total_return': self._calculate_total_return(values),
            'annual_return': self._calculate_annual_return(values, agent_data['timestamp']),
            'volatility': self._calculate_volatility(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns, agent_data['timestamp']),
            'max_drawdown': self._calculate_max_drawdown(values),
            'win_rate': self._calculate_win_rate(returns),
            'current_value': values[-1],
            'num_positions': self._count_positions(agent_data.iloc[-1]['holdings']),
            'trading_days': self._calculate_trading_days(agent_data['timestamp']),
            'start_date': agent_data['trading_date'].min() if 'trading_date' in agent_data.columns else agent_data['timestamp'].min(),
            'end_date': agent_data['trading_date'].max() if 'trading_date' in agent_data.columns else agent_data['timestamp'].max()
        }
    
    def get_all_agent_performance(self) -> pd.DataFrame:
        """获取所有代理的性能指标"""
        results = []
        
        for config_id in self.portfolio_df['config_id'].unique():
            performance = self.get_agent_performance(config_id)
            if performance:
                performance['config_id'] = config_id
                results.append(performance)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('total_return', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_portfolio_evolution(self, config_id: str) -> pd.DataFrame:
        """
        获取投资组合的时间序列演化
        
        Args:
            config_id: 配置ID
            
        Returns:
            包含时间序列数据的DataFrame
        """
        agent_data = self.portfolio_df[self.portfolio_df['config_id'] == config_id].sort_values('timestamp')
        
        if agent_data.empty:
            return pd.DataFrame()
        
        # 计算累计收益率
        values = agent_data['total_value'].values
        cumulative_returns = (values / values[0] - 1) * 100
        
        result_df = agent_data[['timestamp', 'trading_date', 'total_value', 'cashflow', 'holdings']].copy()
        result_df['cumulative_return'] = cumulative_returns
        
        # 计算每日收益率
        daily_returns = np.diff(values) / values[:-1] * 100
        result_df['daily_return'] = [np.nan] + list(daily_returns)
        
        return result_df
    
    def get_holdings_analysis(self, config_id: str = None) -> Dict:
        """
        分析持仓情况
        
        Args:
            config_id: 可选的特定配置ID，如果为None则分析所有配置
            
        Returns:
            包含持仓分析的字典
        """
        if config_id:
            data = self.portfolio_df[self.portfolio_df['config_id'] == config_id]
        else:
            data = self.portfolio_df
        
        # 获取最新的持仓数据
        latest_data = data.groupby('config_id').last().reset_index()
        
        all_tickers = {}
        agent_holdings = {}
        
        for _, row in latest_data.iterrows():
            agent_id = row['config_id']
            agent_holdings[agent_id] = {}
            
            try:
                if row['holdings'] and row['holdings'] != '{}':
                    holdings = json.loads(row['holdings'])
                    for ticker, ticker_data in holdings.items():
                        value = ticker_data.get('value', 0)
                        shares = ticker_data.get('shares', 0)
                        
                        # 累计所有代理的持仓
                        if ticker not in all_tickers:
                            all_tickers[ticker] = {'total_value': 0, 'total_shares': 0, 'agents': 0}
                        
                        all_tickers[ticker]['total_value'] += value
                        all_tickers[ticker]['total_shares'] += shares
                        all_tickers[ticker]['agents'] += 1
                        
                        # 记录单个代理的持仓
                        agent_holdings[agent_id][ticker] = ticker_data
            except:
                continue
        
        return {
            'all_tickers': all_tickers,
            'agent_holdings': agent_holdings,
            'most_popular_ticker': max(all_tickers.items(), key=lambda x: x[1]['agents'])[0] if all_tickers else None,
            'largest_holding': max(all_tickers.items(), key=lambda x: x[1]['total_value'])[0] if all_tickers else None
        }
    
    def get_market_overview(self) -> Dict:
        """获取市场总览数据"""
        performance_df = self.get_all_agent_performance()
        
        if performance_df.empty:
            return {}
        
        holdings_analysis = self.get_holdings_analysis()
        
        return {
            'total_agents': len(performance_df),
            'total_aum': performance_df['current_value'].sum(),
            'average_return': performance_df['total_return'].mean(),
            'best_performer': performance_df.iloc[0]['agent_name'],
            'best_return': performance_df.iloc[0]['total_return'],
            'worst_performer': performance_df.iloc[-1]['agent_name'],
            'worst_return': performance_df.iloc[-1]['total_return'],
            'average_volatility': performance_df['volatility'].mean(),
            'average_sharpe': performance_df['sharpe_ratio'].mean(),
            'total_unique_tickers': len(holdings_analysis['all_tickers']),
            'most_popular_ticker': holdings_analysis['most_popular_ticker'],
            'largest_holding': holdings_analysis['largest_holding']
        }
    
    def compare_agents(self, config_ids: List[str]) -> pd.DataFrame:
        """
        比较多个代理的性能
        
        Args:
            config_ids: 要比较的配置ID列表
            
        Returns:
            比较结果的DataFrame
        """
        comparison_data = []
        
        for config_id in config_ids:
            performance = self.get_agent_performance(config_id)
            if performance:
                comparison_data.append(performance)
        
        return pd.DataFrame(comparison_data)
    
    def _calculate_returns(self, values: np.ndarray) -> np.ndarray:
        """计算收益率序列"""
        if len(values) < 2:
            return np.array([])
        return np.diff(values) / values[:-1]
    
    def _calculate_total_return(self, values: np.ndarray) -> float:
        """计算总收益率"""
        if len(values) < 2 or values[0] == 0:
            return 0.0
        return (values[-1] / values[0] - 1) * 100
    
    def _calculate_annual_return(self, values: np.ndarray, timestamps: pd.Series) -> float:
        """计算年化收益率"""
        if len(values) < 2 or values[0] == 0:
            return 0.0
        
        days = (timestamps.max() - timestamps.min()).days
        if days <= 0:
            return 0.0
        
        return ((values[-1] / values[0]) ** (365 / days) - 1) * 100
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """计算年化波动率"""
        if len(returns) <= 1:
            return 0.0
        return np.std(returns) * np.sqrt(252) * 100
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, timestamps: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) <= 1:
            return 0.0
        
        days = (timestamps.max() - timestamps.min()).days
        if days <= 0:
            return 0.0
        
        annual_return = ((np.prod(1 + returns)) ** (365 / days) - 1) * 100
        volatility = self._calculate_volatility(returns)
        
        if volatility == 0:
            return 0.0
        
        return (annual_return - risk_free_rate * 100) / volatility
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """计算最大回撤"""
        if len(values) < 2:
            return 0.0
        
        returns = self._calculate_returns(values)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown) * 100
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """计算胜率"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns) * 100
    
    def _count_positions(self, holdings_str: str) -> int:
        """计算持仓数量"""
        try:
            if holdings_str and holdings_str != '{}':
                holdings = json.loads(holdings_str)
                return len(holdings)
        except:
            pass
        return 0
    
    def _calculate_trading_days(self, timestamps: pd.Series) -> int:
        """计算交易天数"""
        return (timestamps.max() - timestamps.min()).days

class PerformanceComparer:
    """性能比较器类"""
    
    def __init__(self, analyzer: DataAnalyzer):
        self.analyzer = analyzer
    
    def rank_by_metric(self, metric: str) -> pd.DataFrame:
        """根据指定指标排名"""
        performance_df = self.analyzer.get_all_agent_performance()
        
        if performance_df.empty or metric not in performance_df.columns:
            return pd.DataFrame()
        
        # 根据指标排序（大部分指标越大越好，除了波动率和最大回撤）
        ascending = metric in ['volatility', 'max_drawdown']
        ranked_df = performance_df.sort_values(metric, ascending=ascending)
        ranked_df[f'{metric}_rank'] = range(1, len(ranked_df) + 1)
        
        return ranked_df
    
    def get_top_performers(self, n: int = 5, metric: str = 'total_return') -> pd.DataFrame:
        """获取顶级表现者"""
        ranked_df = self.rank_by_metric(metric)
        return ranked_df.head(n)
    
    def get_risk_adjusted_ranking(self) -> pd.DataFrame:
        """获取风险调整后的排名"""
        return self.rank_by_metric('sharpe_ratio')

def load_and_analyze_data(config_path: str = 'data/config_rows_new.csv', 
                         portfolio_path: str = 'data/portfolio_rows_new.csv') -> DataAnalyzer:
    """
    加载数据并创建分析器
    
    Args:
        config_path: 配置文件路径
        portfolio_path: 投资组合文件路径
        
    Returns:
        DataAnalyzer实例
    """
    try:
        config_df = pd.read_csv(config_path)
        portfolio_df = pd.read_csv(portfolio_path)
        
        # 重命名字段以保持兼容性
        portfolio_df = portfolio_df.rename(columns={
            'updated_at': 'timestamp',
            'total_assets': 'total_value',
            'positions': 'holdings'
        })
        
        return DataAnalyzer(config_df, portfolio_df)
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None 