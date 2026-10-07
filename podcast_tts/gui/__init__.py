"""Podvia Windows GUI（阶段 6，Tkinter）。

分层：
- settings.py：GUI 设置数据、校验（全部复用 capabilities/config，不另立规则）、持久化
- controller.py：后台线程 + queue 事件（Tk 主线程绝不直接调用 pipeline/网络）
- app.py：Tkinter 控件与 after() 轮询；动态控件依据能力矩阵
"""
