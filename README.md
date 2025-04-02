# Deepfund: The most advanced LLM finance arena

这是一个基于Streamlit构建的先进金融交易平台界面，专注于LLM（大型语言模型）驱动的交易策略和工作流。

## 功能特点

- 导航栏带有搜索功能
- Top 10整体工作流展示
- 可选择不同工作流查看其性能
- 支持折线图和蜡烛图切换显示
- 可选择不同时间周期(1D, 1M, 3M, 1Y, 5Y, All)
- 性能指标展示(夏普比率、最大回撤等)
- 最近交易记录展示
- 响应式UI，模仿专业交易平台设计

## 安装说明

1. 克隆此仓库
2. 安装所需包:
   ```
   pip install -r requirements.txt
   ```

## 运行应用

使用以下命令运行Streamlit应用:

```
streamlit run streamlit_app.py
```

应用将在浏览器中打开，默认地址为 http://localhost:8501

## 数据来源

目前，应用使用随机生成的模拟数据用于演示目的。在生产环境中，您可以用真实市场数据替换，例如:

- Alpha Vantage
- Yahoo Finance
- IEX Cloud
- Finnhub
- Polygon.io

## 自定义设置

您可以通过修改`streamlit_app.py`文件自定义工作流、图表外观和其他UI元素。
