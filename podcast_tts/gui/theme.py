"""Podvia GUI 设计 Token：颜色 / 字体 / 间距，唯一事实来源。

app.py 与弹窗统一从这里取色值与字体，不再散落硬编码十六进制。
主题遵循 Flat Design：中性灰底 + 白色卡片 + 单一品牌强调色。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# —— 颜色 ——————————————————————————————————————————————————————————————
BG = "#FAFAFA"            # 窗口底色（中性灰）
CARD = "#FFFFFF"          # 卡片 / 内容区背景
PRIMARY = "#0D9488"       # 品牌主色（teal）
PRIMARY_FG = "#FFFFFF"    # 主色上的文字
ACCENT = "#EA580C"        # 强调色（仅"生成"主按钮）
ACCENT_ACTIVE = "#C2410C"
FG = "#134E4A"            # 主文字
MUTED_FG = "#475569"      # 次要文字 / 说明
BORDER = "#E5E7EB"        # 分隔线 / 边框
NAV_ACTIVE_BG = "#0D9488"
NAV_ACTIVE_FG = "#FFFFFF"
NAV_BG = "#F1F5F9"

# 状态色（统一全应用；替换原 #0a7a30 / #b36b00 / #c00 / #666 等散落值）
STATE_OK = "#0A7A30"      # 已生成 / 有效
STATE_DIRTY = "#B36B00"   # 待更新
STATE_ERROR = "#DC2626"   # 失败 / 需重新选择
STATE_MISSING = "#475569"  # 未生成 / 未核实
STATE_PAUSE = "#2563EB"   # 停顿
STATE_CUSTOM = "#8A5A00"  # custom 音色提示

# Treeview 行底色（片段管理 / 角色状态）
ROW_DIRTY = "#FFF4E0"
ROW_FAILED = "#FFE9E9"
ROW_MISSING = "#F0F0F0"
ROW_PAUSE = "#EEF6FF"

# —— 字体 ——————————————————————————————————————————————————————————————
FONT_FAMILY = ("Segoe UI", "Microsoft YaHei UI")
FONT_MONO = ("Consolas",)
FONT_SIZE = 9
FONT_BASE = (FONT_FAMILY, FONT_SIZE)
FONT_BOLD = (FONT_FAMILY, FONT_SIZE, "bold")
FONT_TITLE = (FONT_FAMILY, 12, "bold")
FONT_EDITOR = (FONT_MONO, 10)

# —— 间距（8pt 网格） ——————————————————————————————————————————————————
PAD_XS = 4
PAD_SM = 8
PAD_MD = 12
PAD_LG = 16


def apply_theme(root: tk.Misc) -> ttk.Style:
    """配置 ttk 样式（在创建任何控件前调用一次）。返回 Style 对象。"""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=FONT_BASE, background=BG, foreground=FG)
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD)
    style.configure("Nav.TFrame", background=NAV_BG)
    style.configure("TLabel", background=BG, foreground=FG)
    style.configure("Card.TLabel", background=CARD, foreground=FG)
    style.configure("Muted.TLabel", foreground=MUTED_FG)
    style.configure("CardMuted.TLabel", background=CARD, foreground=MUTED_FG)
    style.configure("Title.TLabel", font=FONT_TITLE, foreground=FG)
    style.configure("TLabelframe", background=BG, foreground=FG)
    style.configure("TLabelframe.Label", background=BG, foreground=FG)
    style.configure("TButton", padding=(PAD_MD, PAD_XS))
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.configure("TRadiobutton", background=BG, foreground=FG)
    style.configure("TMenubutton", padding=(PAD_MD, PAD_XS))

    # 主按钮（生成 / 更新音频）——唯一使用 Accent 的控件
    style.configure(
        "Accent.TButton", background=ACCENT, foreground=PRIMARY_FG,
        padding=(PAD_LG, PAD_SM), font=FONT_BOLD, borderwidth=0)
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_ACTIVE), ("disabled", "#D1D5DB")],
        foreground=[("disabled", "#6B7280")])

    # 左侧导航按钮
    style.configure(
        "Nav.TButton", background=NAV_BG, foreground=FG, borderwidth=0,
        padding=(PAD_MD, PAD_SM), anchor="w", font=FONT_BASE)
    style.map(
        "Nav.TButton",
        background=[("active", BORDER)],
        foreground=[("active", FG)])
    style.configure(
        "NavActive.TButton", background=NAV_ACTIVE_BG, foreground=NAV_ACTIVE_FG,
        borderwidth=0, padding=(PAD_MD, PAD_SM), anchor="w", font=FONT_BOLD)
    style.map(
        "NavActive.TButton",
        background=[("active", PRIMARY)],
        foreground=[("active", PRIMARY_FG)])

    # Treeview
    style.configure(
        "Treeview", background=CARD, fieldbackground=CARD, foreground=FG,
        rowheight=24, bordercolor=BORDER)
    style.configure(
        "Treeview.Heading", background=NAV_BG, foreground=FG, font=FONT_BOLD)
    style.map("Treeview", background=[("selected", PRIMARY)],
              foreground=[("selected", PRIMARY_FG)])

    return style
