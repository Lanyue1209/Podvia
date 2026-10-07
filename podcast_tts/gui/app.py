"""Podvia Windows GUI（通用语音合成工具，Tkinter）。

定位：Podvia 是通用语音合成工具——文本、角色、音色、语言、语音参数、导演指令、
SSML、输出格式与目录是主界面的核心；章节文件导入只是输入方式之一。

纪律：
- 所有业务动作走 GuiController → pipeline；GUI 不做 TTS/缓存/合并。
- 工作线程只投递事件；after() 轮询 queue，控件更新只在主线程。
- 角色由用户动态管理（稳定 id 索引）；模型切换后失效音色只标记、不静默替换。
- API Key 明文只在内存；落盘经 DPAPI（settings.save），日志只打印脱敏值。
"""
from __future__ import annotations

import os
import shutil
import tkinter as tk
import tkinter.simpledialog as simpledialog
from dataclasses import replace
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from podcast_tts import capabilities as caps
from podcast_tts import emotags
from podcast_tts.config import ConfigError
from podcast_tts.gui import secret_store
from podcast_tts.gui import theme
from podcast_tts.gui.controller import (
    GuiController,
    chapter_summary,
    preview_summary,
    preview_working,
)
from podcast_tts.gui.settings import (
    GuiSettings,
    RoleVoice,
    default_settings_path,
    settings_from_config,
)
from podcast_tts.project import WorkingScript

SAMPLE_RATES = [8000, 12000, 16000, 22050, 24000, 44100, 48000]
LANGUAGE_HINTS = ["", "zh", "en", "fr", "de", "ja", "ko", "ru", "pt", "th",
                  "id", "vi", "es", "it", "ms", "fil", "ar"]

# 需求约定的四档状态文案（逐字）
STATUS_TEXT = {
    "idle": "空闲",
    "running": "正在生成片段…",
    "paused": "已暂停",
    "cancelling": "正在取消…",
    "merging": "正在拼接…",
    "done": "已完成",
    "merge_failed": "片段生成完成但拼接失败",
    "cancelled": "已取消",
    "error": "出错",
}
_BUSY_STATES = {"running", "paused", "cancelling", "merging"}

_STATUS_STYLE = {
    "ok": ("有效", theme.STATE_OK),
    "custom": ("自定义音色（云端存在性无法离线验证）", theme.STATE_CUSTOM),
    "unknown": ("未核实（将按基线发送）", theme.STATE_MISSING),
    "invalid": ("需重新选择（不属于当前模型）", theme.STATE_ERROR),
    "empty": ("未选择音色", theme.STATE_ERROR),
}

# 创作页文本输入框的一行占位提示（完整语法见"语法帮助"弹窗）
PLACEHOLDER_TEXT = (
    "粘贴文本或导入章节文件。写法：角色名：台词；停顿独占一行 [PAUSE:1.5]；"
    "完整语法见右下角「语法帮助」。"
)

# 语法帮助弹窗全文（由原占位教程与 SSML/优先级说明汇总而来）
HELP_TEXT = """\
脚本语法
========

写法一：每段以「角色名：台词」开头（角色名需已在「角色」页注册，
可用中文；也支持 [旁白] 台词）。

写法二：不写角色名，但必须在「设置」页选择「默认角色（可选）」；
留空时无角色标记会直接报错，不会静默选择角色。

行内指令与参数示例：
  Lily(轻快地)：台词
  Lily(轻快地; rate=1.2)：台词

停顿独占一行写 [PAUSE:1.5]（秒，0-10），可出现在脚本开头、
两段台词之间或结尾。

单段超过 480 字会在生成前自动按句子安全切分（阿里云 API 硬限制
600 字，标签也占用长度）。

情绪与人声动作标签（如 [excited]、[laughing]）仅限已确认支持的
模型（qwen-audio-3.1-tts-flash / 3.0-tts-plus / 3.0-tts-flash）；
环境音效/背景音乐标记（如【音效：脚步声】、[BGM:…]）不支持，
出现会直接报错。

SSML
====

启用 SSML 后，每段必须是合法 SSML（以 <speak> 为根节点），
普通文本不会被自动改写，且不能再使用情绪标签。

白名单：speak(rate/pitch/volume) / break / phoneme / sub / say-as；
不支持 prosody / emphasis。
break 标签必须自闭合，time 属性为整数（1-10s 或 50-10000ms）。

优先级（后者覆盖前者，每段只取最终一条指令/一组参数）
=====================================================

导演指令：项目默认 < 角色默认 < 章节（章节导入） < 行内
语速/音调/音量：项目默认 < 角色默认 < 行内片段
SSML <speak> 标签属性优先级最高（出现时不再发送同名 HTTP 参数）。

修改任意层级参数或指令都会进入缓存 hash，受影响片段会自动重新生成；
当前模型不支持的细粒度参数会直接报错，不会静默忽略。
"""


class PodviaApp(tk.Tk):
    def __init__(self, controller: GuiController | None = None):
        super().__init__()
        self.title("Podvia — 通用语音合成（Windows）")
        self.geometry("1180x860")
        self.minsize(880, 560)
        self.configure(background=theme.BG)
        theme.apply_theme(self)
        self.controller = controller or GuiController()
        self.settings = self._initial_settings()
        self._failed: list[dict] = []
        self._failed_merge: dict | None = None
        self._save_after_id: str | None = None
        # 当前工作脚本（粘贴/导入统一形态）；None = 尚未导入/生成
        self.ws: WorkingScript | None = None
        self._text_dirty = False  # 文本框是否被手动改动过（重新导入保护用）
        self._placeholder_active = False
        self._log_expanded = False
        self._build_widgets()
        self._sync_model_choices()
        self._sync_settings_to_widgets()
        self.after(120, self._poll_events)
        self.after(300, self._log_ffmpeg_hint)
        self.after(400, self._refresh_summary)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # —— 初始化 ————————————————————————————————————————————————————————

    def _initial_settings(self) -> GuiSettings:
        loaded = GuiSettings.load(default_settings_path())
        return loaded or settings_from_config()

    def _build_widgets(self) -> None:
        """三页左导航布局：创作（主工作区） / 角色（整页表格） / 设置。

        纪律：所有业务动作仍走 GuiController；这里只做控件摆放与状态展示。
        """
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        # —— 左侧导航 ——————————————————————————————————————————————————
        nav = ttk.Frame(outer, style="Nav.TFrame", width=112)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)
        ttk.Label(nav, text="Podvia", font=theme.FONT_TITLE,
                  background=theme.NAV_BG).pack(
                      padx=theme.PAD_MD, pady=(theme.PAD_LG, theme.PAD_MD),
                      anchor="w")
        self._nav_buttons: dict[str, ttk.Button] = {}
        for key, label in (("create", "创 作"), ("roles", "角 色"),
                           ("settings", "设 置")):
            b = ttk.Button(nav, text=label, style="Nav.TButton",
                           command=lambda k=key: self._show_page(k))
            b.pack(fill="x", padx=theme.PAD_SM, pady=2)
            self._nav_buttons[key] = b

        # —— 右侧内容区：三个页面 Frame 叠放，导航切换 ———————————————————
        container = ttk.Frame(outer)
        container.pack(side="left", fill="both", expand=True)
        self.page_create = ttk.Frame(container)
        self.page_roles = ttk.Frame(container)
        self.page_settings = ttk.Frame(container)
        for p in (self.page_create, self.page_roles, self.page_settings):
            p.place(relwidth=1, relheight=1)

        self._build_create_page()
        self._build_roles_page()
        self._build_settings_page()
        self._show_page("create")

        # 非敏感控件改动 → 防抖自动保存
        for var in (self.var_provider, self.var_model, self.var_rate,
                    self.var_pitch, self.var_volume, self.var_seed,
                    self.var_sample_rate, self.var_format, self.var_direction,
                    self.var_episode, self.var_outdir, self.var_default_role):
            var.trace_add("write", lambda *_: self._schedule_save())
        self.cmb_sr.bind("<<ComboboxSelected>>", lambda e: self._schedule_save())
        self.cmb_format.bind("<<ComboboxSelected>>", lambda e: self._schedule_save())

    def _show_page(self, key: str) -> None:
        pages = {"create": self.page_create, "roles": self.page_roles,
                 "settings": self.page_settings}
        self._current_page = key
        for k, p in pages.items():
            if k == key:
                p.lift()
                self._nav_buttons[k].configure(style="NavActive.TButton")
            else:
                self._nav_buttons[k].configure(style="Nav.TButton")

    # —— 创作页 ————————————————————————————————————————————————————————

    def _build_create_page(self) -> None:
        page = self.page_create

        # 顶部：episode 名 + 来源标签
        top = ttk.Frame(page)
        top.pack(fill="x", padx=theme.PAD_LG, pady=(theme.PAD_LG, theme.PAD_SM))
        ttk.Label(top, text="episode 名").pack(side="left")
        self.var_episode = tk.StringVar(value="episode")
        ttk.Entry(top, textvariable=self.var_episode, width=18).pack(
            side="left", padx=theme.PAD_SM)
        self.var_book_path = tk.StringVar()
        self.lbl_book = ttk.Label(top, text="来源：直接粘贴",
                                  style="Muted.TLabel")
        self.lbl_book.pack(side="left", padx=(theme.PAD_MD, 2))
        self.btn_clear_book = ttk.Button(
            top, text="✕", width=3, command=self._clear_book, state="disabled")
        self.btn_clear_book.pack(side="left")
        ttk.Button(top, text="导入章节文件…",
                   command=self._pick_book).pack(side="right")

        # 文本编辑区（主角，垂直撑满）
        editor_card = ttk.Frame(page, style="Card.TFrame")
        editor_card.pack(fill="both", expand=True,
                         padx=theme.PAD_LG, pady=theme.PAD_SM)
        self.txt_input = tk.Text(
            editor_card, wrap="word", undo=True, font=theme.FONT_EDITOR,
            background=theme.CARD, foreground=theme.FG,
            insertbackground=theme.FG, relief="flat",
            highlightthickness=1, highlightcolor=theme.BORDER,
            highlightbackground=theme.BORDER)
        txt_sb = ttk.Scrollbar(editor_card, orient="vertical",
                               command=self.txt_input.yview)
        self.txt_input.configure(yscrollcommand=txt_sb.set)
        self.txt_input.pack(side="left", fill="both", expand=True,
                            padx=(theme.PAD_MD, 0), pady=theme.PAD_MD)
        txt_sb.pack(side="right", fill="y", pady=theme.PAD_MD,
                    padx=(0, theme.PAD_XS))
        self._init_placeholder()
        self.txt_input.bind("<FocusIn>", self._placeholder_focus_in)
        self.txt_input.bind("<FocusOut>", self._placeholder_focus_out)
        # 手动改动文本 → 标记（导入填充后会复位，用于重新导入保护）
        self.txt_input.bind("<<Modified>>", self._on_text_modified)

        # 编辑区下方工具条：插入标签 + 语法帮助
        tag_bar = ttk.Frame(page)
        tag_bar.pack(fill="x", padx=theme.PAD_LG)
        ttk.Label(tag_bar, text="插入标签：").pack(side="left")
        self.mbtn_emo = ttk.Menubutton(tag_bar, text="情绪（作用于后续文字）")
        menu_emo = tk.Menu(self.mbtn_emo, tearoff=0)
        for tag, zh in emotags.CONTROL_TAGS.items():
            menu_emo.add_command(
                label=f"{zh} {tag}", command=lambda t=tag: self._insert_tag(t))
        self.mbtn_emo.config(menu=menu_emo)
        self.mbtn_emo.pack(side="left", padx=theme.PAD_XS)
        self.mbtn_vocal = ttk.Menubutton(tag_bar, text="人声动作（一次性拟声）")
        menu_vocal = tk.Menu(self.mbtn_vocal, tearoff=0)
        for tag, zh in emotags.PARALINGUISTIC_TAGS.items():
            menu_vocal.add_command(
                label=f"{zh} {tag}", command=lambda t=tag: self._insert_tag(t))
        self.mbtn_vocal.config(menu=menu_vocal)
        self.mbtn_vocal.pack(side="left", padx=theme.PAD_XS)
        self.lbl_tag_support = ttk.Label(tag_bar, text="",
                                         foreground=theme.STATE_ERROR)
        self.lbl_tag_support.pack(side="left", padx=theme.PAD_SM)
        ttk.Button(tag_bar, text="语法帮助",
                   command=self._show_help).pack(side="right")

        # 状态区（集中）：摘要 + 进度 + 状态 + 可折叠日志
        status = ttk.Frame(page)
        status.pack(fill="x", padx=theme.PAD_LG, pady=theme.PAD_SM)
        self.lbl_summary = ttk.Label(
            status, text="", foreground=theme.FG,
            justify="left", wraplength=900)
        self.lbl_summary.pack(fill="x")
        prog_line = ttk.Frame(status)
        prog_line.pack(fill="x", pady=(theme.PAD_XS, 0))
        self.progress = ttk.Progressbar(prog_line, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.lbl_progress = ttk.Label(prog_line, text=STATUS_TEXT["idle"],
                                      width=22)
        self.lbl_progress.pack(side="left", padx=theme.PAD_SM)
        self.btn_toggle_log = ttk.Button(status, text="查看日志 ▸",
                                         command=self._toggle_log)
        self.btn_toggle_log.pack(anchor="w", pady=(theme.PAD_XS, 0))
        self.log_frame = ttk.Frame(status, style="Card.TFrame")
        self.txt_log = tk.Text(
            self.log_frame, height=6, wrap="word", state="disabled",
            font=theme.FONT_EDITOR, background=theme.CARD,
            foreground=theme.MUTED_FG, relief="flat")
        self.txt_log.pack(fill="x", padx=theme.PAD_SM, pady=theme.PAD_SM)
        # 默认折叠，不 pack log_frame

        # 操作行（固定底部）
        ctl = ttk.Frame(page)
        ctl.pack(fill="x", padx=theme.PAD_LG,
                 pady=(0, theme.PAD_LG))
        self.btn_start = ttk.Button(ctl, text="生成 / 更新音频",
                                    style="Accent.TButton",
                                    command=self._on_start)
        self.btn_start.pack(side="left")
        self.btn_pause = ttk.Button(ctl, text="暂停",
                                    command=self._on_pause_resume_toggle,
                                    state="disabled")
        self.btn_pause.pack(side="left", padx=theme.PAD_SM)
        self.btn_cancel = ttk.Button(ctl, text="取消",
                                     command=self.controller.cancel,
                                     state="disabled")
        self.btn_cancel.pack(side="left")
        self.btn_manage = ttk.Button(ctl, text="片段管理…",
                                     command=self._on_manage_segments)
        self.btn_manage.pack(side="left", padx=theme.PAD_SM)
        self.btn_export = ttk.Menubutton(ctl, text="导出 ▾")
        menu_exp = tk.Menu(self.btn_export, tearoff=0)
        menu_exp.add_command(label="播放成品（整章）",
                             command=self._on_play_chapter)
        menu_exp.add_command(label="导出字幕（SRT）…",
                             command=self._on_export_srt)
        self.btn_export.config(menu=menu_exp)
        self.btn_export.pack(side="left")
        self.mbtn_advanced = ttk.Menubutton(ctl, text="⋯")
        menu_adv = tk.Menu(self.mbtn_advanced, tearoff=0)
        menu_adv.add_command(label="生成所有未完成片段（失败/未生成/待更新）",
                             command=self._on_retry_failed)
        menu_adv.add_command(label="局部重生成（按片段编号）…",
                             command=self._on_regenerate)
        menu_adv.add_command(label="仅重新合并（使用现有片段文件，可能含旧音频）",
                             command=self._on_merge_only)
        self.mbtn_advanced.config(menu=menu_adv)
        self._menu_advanced = menu_adv
        self.mbtn_advanced.pack(side="left", padx=theme.PAD_SM)

    # —— 占位文本（1 行） ———————————————————————————————————————————————

    def _init_placeholder(self) -> None:
        self._placeholder_active = True
        self.txt_input.insert("1.0", PLACEHOLDER_TEXT)
        self.txt_input.config(foreground=theme.MUTED_FG)
        self.txt_input.edit_modified(False)  # 复位占位插入产生的 modified

    def _clear_placeholder(self) -> None:
        if self._placeholder_active:
            self._placeholder_active = False
            self.txt_input.delete("1.0", "end")
            self.txt_input.config(foreground=theme.FG)

    def _placeholder_focus_in(self, _evt=None) -> None:
        self._clear_placeholder()

    def _placeholder_focus_out(self, _evt=None) -> None:
        if not self.txt_input.get("1.0", "end").strip():
            self._init_placeholder()

    def _show_help(self) -> None:
        win = tk.Toplevel(self)
        win.title("语法帮助")
        win.geometry("640x560")
        win.transient(self)
        txt = tk.Text(win, wrap="word", font=theme.FONT_BASE,
                      background=theme.CARD, foreground=theme.FG,
                      relief="flat", padx=theme.PAD_MD, pady=theme.PAD_MD)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", HELP_TEXT)
        txt.config(state="disabled")
        ttk.Button(win, text="关闭", command=win.destroy).pack(
            pady=theme.PAD_SM)

    def _toggle_log(self) -> None:
        if self._log_expanded:
            self.log_frame.pack_forget()
            self.btn_toggle_log.config(text="查看日志 ▸")
            self._log_expanded = False
        else:
            self.log_frame.pack(fill="x", pady=(theme.PAD_XS, 0),
                                before=self.btn_toggle_log)
            self.btn_toggle_log.config(text="收起日志 ▾")
            self._log_expanded = True

    def _on_pause_resume_toggle(self) -> None:
        """暂停/继续合并按钮：依据当前控制器状态分流。"""
        if self.controller.state == "running":
            self.controller.pause()
        elif self.controller.state == "paused":
            self.controller.resume()

    # —— 角色页 ————————————————————————————————————————————————————————

    def _build_roles_page(self) -> None:
        page = self.page_roles

        # 页顶：模型只读提示 + 统计
        top = ttk.Frame(page)
        top.pack(fill="x", padx=theme.PAD_LG, pady=(theme.PAD_LG, theme.PAD_SM))
        self.lbl_model_status = ttk.Label(top, text="", style="Muted.TLabel")
        self.lbl_model_status.pack(side="left")
        self.lbl_counts = ttk.Label(top, text="", foreground=theme.STATE_OK)
        self.lbl_counts.pack(side="left", padx=theme.PAD_MD)
        self.lbl_reuse = ttk.Label(top, text="",
                                   foreground=theme.STATE_DIRTY)
        self.lbl_reuse.pack(side="left")

        # 表头
        header = ttk.Frame(page)
        header.pack(fill="x", padx=theme.PAD_LG)
        for col, text in enumerate((
                "角色名称（可中文）", "音色 ID（可输入 custom ID）", "语言",
                "custom", "角色默认导演指令",
                "默认语速", "默认音调", "默认音量", "状态", "", "")):
            ttk.Label(header, text=text, style="Muted.TLabel").grid(
                row=0, column=col, padx=theme.PAD_XS, sticky="w")

        # 角色行容器（整页可拉伸，无内层 canvas）
        self.role_frame = ttk.Frame(page)
        self.role_frame.pack(fill="both", expand=True,
                             padx=theme.PAD_LG, pady=theme.PAD_SM)
        self._role_rows: dict[str, dict] = {}

        # 底部操作
        ctl = ttk.Frame(page)
        ctl.pack(fill="x", padx=theme.PAD_LG, pady=(0, theme.PAD_LG))
        ttk.Button(ctl, text="添加角色", command=self._on_add_role).pack(
            side="left")
        self.var_distinct = tk.BooleanVar(value=True)
        self.chk_distinct = ttk.Checkbutton(
            ctl, text="每个角色使用不同音色", variable=self.var_distinct,
            command=self._on_distinct_toggle)
        self.chk_distinct.pack(side="left", padx=theme.PAD_MD)

    # —— 设置页 ————————————————————————————————————————————————————————

    def _build_settings_page(self) -> None:
        page = self.page_settings
        pad = {"padx": theme.PAD_LG, "pady": theme.PAD_XS}

        # API Key
        bar = ttk.LabelFrame(page, text="API Key（点击开始或关闭窗口时自动保存到本机，DPAPI 加密）")
        bar.pack(fill="x", padx=theme.PAD_LG, pady=(theme.PAD_LG, theme.PAD_SM))
        self.var_key = tk.StringVar()
        ent_key = ttk.Entry(bar, textvariable=self.var_key, show="*", width=48)
        ent_key.pack(side="left", **pad)
        ent_key.bind("<KeyRelease>", lambda e: self._schedule_save())
        ttk.Label(bar, text="存放位置：%APPDATA%/Podvia/gui_settings.json；"
                            "不会进入 .env / config.yaml / manifest / 安装包",
                  style="Muted.TLabel").pack(side="left")

        # Provider / 模型
        mbox = ttk.LabelFrame(page, text="Provider / 模型")
        mbox.pack(fill="x", padx=theme.PAD_LG, pady=theme.PAD_SM)
        ttk.Label(mbox, text="Provider").grid(row=0, column=0, sticky="w", **pad)
        self.var_provider = tk.StringVar()
        self.cmb_provider = ttk.Combobox(
            mbox, textvariable=self.var_provider, width=16, state="readonly",
            values=["qwen_audio", "qwen"])
        self.cmb_provider.grid(row=0, column=1, sticky="w")
        self.cmb_provider.bind("<<ComboboxSelected>>",
                               lambda e: self._sync_model_choices())
        ttk.Label(mbox, text="模型").grid(row=0, column=2, sticky="w", **pad)
        self.var_model = tk.StringVar()
        self.cmb_model = ttk.Combobox(mbox, textvariable=self.var_model,
                                      width=32, state="readonly")
        self.cmb_model.grid(row=0, column=3, sticky="w")
        self.cmb_model.bind("<<ComboboxSelected>>",
                            lambda e: self._on_model_change())

        # 全局语音参数
        pbox = ttk.LabelFrame(page, text="全局语音参数")
        pbox.pack(fill="x", padx=theme.PAD_LG, pady=theme.PAD_SM)
        self.var_rate = tk.StringVar()
        self.var_pitch = tk.StringVar()
        self.var_volume = tk.StringVar()
        self.var_seed = tk.StringVar()
        self.var_sample_rate = tk.StringVar(value="24000")
        self.var_format = tk.StringVar(value="wav")
        ttk.Label(pbox, text="语速(0.5-2.0)").grid(row=0, column=0, sticky="e", **pad)
        self.ent_rate = ttk.Entry(pbox, textvariable=self.var_rate, width=8)
        self.ent_rate.grid(row=0, column=1)
        ttk.Label(pbox, text="音调(0.5-2.0)").grid(row=0, column=2, sticky="e", **pad)
        self.ent_pitch = ttk.Entry(pbox, textvariable=self.var_pitch, width=8)
        self.ent_pitch.grid(row=0, column=3)
        ttk.Label(pbox, text="音量(0-100)").grid(row=0, column=4, sticky="e", **pad)
        self.ent_volume = ttk.Entry(pbox, textvariable=self.var_volume, width=8)
        self.ent_volume.grid(row=0, column=5)
        ttk.Label(pbox, text="seed(0-65535)").grid(row=0, column=6, sticky="e", **pad)
        self.ent_seed = ttk.Entry(pbox, textvariable=self.var_seed, width=10)
        self.ent_seed.grid(row=0, column=7)
        ttk.Label(pbox, text="采样率").grid(row=1, column=0, sticky="e", **pad)
        self.cmb_sr = ttk.Combobox(pbox, textvariable=self.var_sample_rate,
                                   width=8, state="readonly",
                                   values=[str(x) for x in SAMPLE_RATES])
        self.cmb_sr.grid(row=1, column=1)
        ttk.Label(pbox, text="输出格式").grid(row=1, column=2, sticky="e", **pad)
        self.cmb_format = ttk.Combobox(pbox, textvariable=self.var_format,
                                       width=6, state="readonly",
                                       values=["wav", "mp3"])
        self.cmb_format.grid(row=1, column=3)

        # 导演指令 / 默认角色 / SSML
        dbox = ttk.LabelFrame(page, text="导演指令 / 默认角色 / SSML")
        dbox.pack(fill="x", padx=theme.PAD_LG, pady=theme.PAD_SM)
        self.var_direction = tk.StringVar()
        self.var_ssml = tk.BooleanVar(value=False)
        self.var_default_role = tk.StringVar()
        ttk.Label(dbox, text="默认导演指令（项目级，对未单独设置指令的片段生效）").grid(
            row=0, column=0, sticky="e", **pad)
        ttk.Entry(dbox, textvariable=self.var_direction, width=40).grid(
            row=0, column=1, columnspan=3, sticky="we", padx=theme.PAD_XS)
        ttk.Label(dbox, text="默认角色（可选）").grid(row=1, column=0, sticky="e", **pad)
        self.cmb_default_role = ttk.Combobox(
            dbox, textvariable=self.var_default_role, width=16, state="readonly")
        self.cmb_default_role.grid(row=1, column=1, sticky="w")
        ttk.Label(dbox,
                  text="仅对无角色标记的普通文本生效；留空时无角色标记会直接报错",
                  style="Muted.TLabel").grid(
                      row=1, column=2, columnspan=2, sticky="w")
        self.chk_ssml = ttk.Checkbutton(
            dbox, text="启用 SSML（原文必须是合法 SSML，详见创作页「语法帮助」）",
            variable=self.var_ssml, command=self._schedule_save)
        self.chk_ssml.grid(row=2, column=0, columnspan=4, sticky="w",
                           padx=theme.PAD_LG, pady=theme.PAD_XS)

        # 输出目录
        obox = ttk.LabelFrame(page, text="输出目录")
        obox.pack(fill="x", padx=theme.PAD_LG, pady=theme.PAD_SM)
        self.var_outdir = tk.StringVar()
        ttk.Entry(obox, textvariable=self.var_outdir, width=52).pack(
            side="left", padx=theme.PAD_LG, pady=theme.PAD_SM)
        ttk.Button(obox, text="浏览…", command=self._pick_outdir).pack(
            side="left", pady=theme.PAD_SM)
        ttk.Label(obox, text="留空 = 沿用 config.yaml 的 projects_dir",
                  style="Muted.TLabel").pack(side="left", padx=theme.PAD_MD)

    # —— 模型 / 角色动态控件 ————————————————————————————————————————————

    def _profiles(self):
        return caps.list_model_profiles()

    def _current_profile(self) -> dict | None:
        return next((p for p in self._profiles()
                     if p.get("provider") == self.var_provider.get()
                     and p.get("model") == self.var_model.get()), None)

    def _sync_model_choices(self) -> None:
        provider = self.var_provider.get() or self.settings.provider
        models = [p["model"] for p in self._profiles()
                  if p.get("provider") == provider and p["status"] == "implemented"]
        self.cmb_model["values"] = models
        if self.var_model.get() not in models:
            self.var_model.set(self.settings.model if self.settings.model in models
                               else (models[0] if models else ""))
        self.var_provider.set(provider)
        self._sync_settings_provider_model()
        self._on_model_change()

    def _sync_settings_provider_model(self) -> None:
        """把控件当前 provider/model 写入 settings（角色行仍在，不重建）。"""
        self.settings.provider = self.var_provider.get()
        self.settings.model = self.var_model.get()

    def _on_model_change(self) -> None:
        self._collect_from_rows()
        self._sync_settings_provider_model()
        profile = self._current_profile()
        status = caps.describe_model(self.var_model.get()) if self.var_model.get() else ""
        self.lbl_model_status.config(
            text=f"当前模型：{self.var_model.get()}（{status} · 在「设置」页修改）")
        supports = (profile or {}).get("supports", {})
        state_on, state_off = "normal", "disabled"
        for w, key in ((self.ent_rate, "rate"), (self.ent_pitch, "pitch"),
                       (self.ent_volume, "volume"), (self.ent_seed, "seed")):
            w.config(state=state_on if supports.get(key) else state_off)
        self.chk_ssml.config(state=state_on if supports.get("ssml") else state_off)
        if not supports.get("ssml"):
            self.var_ssml.set(False)
        # 情绪/拟声标签入口随模型能力变化；切换模型后重新检查文本里已有的标签
        emotags_ok = bool(supports.get("emotags"))
        for w in (self.mbtn_emo, self.mbtn_vocal):
            w.config(state=state_on if emotags_ok else state_off)
        if emotags_ok:
            self.lbl_tag_support.config(text="")
        else:
            self.lbl_tag_support.config(
                text="当前模型未确认支持情绪/拟声标签（仅 Qwen-Audio-TTS 三个已核实模型）")
            body = self.txt_input.get("1.0", "end") if hasattr(self, "txt_input") else ""
            if emotags.has_tags(body):
                self._log(
                    "提示：文本中包含情绪/拟声标签，但当前模型未确认支持该能力；"
                    "生成前会被阻止，请更换模型或删除标签。"
                )
        self._refresh_voice_choices()
        self._refresh_counts()
        self._sync_role_param_states()
        self._schedule_save()

    def _sync_role_param_states(self) -> None:
        """角色级 rate/pitch/volume 输入框随当前模型能力启用/禁用（值保留不清空）。"""
        supports = (self._current_profile() or {}).get("supports", {})
        for row in self._role_rows.values():
            for key in ("rate", "pitch", "volume"):
                row[f"{key}_widget"].config(
                    state="normal" if supports.get(key) else "disabled")

    def _system_voices(self) -> list[str]:
        profile = self._current_profile()
        return sorted(profile.get("voices", [])) if profile else []

    def _refresh_voice_choices(self) -> None:
        """只刷新下拉候选与状态标记；绝不改写用户已选（失效）音色。"""
        voices = self._system_voices()
        for rv in self.settings.roles:
            row = self._role_rows.get(rv.id)
            if row is None:
                continue
            cmb = row["voice_widget"]
            cmb["values"] = voices
            status = self.settings.role_voice_status(rv)
            text, color = _STATUS_STYLE.get(status, ("", theme.STATE_MISSING))
            row["status"].config(text=text, foreground=color)

    def _rebuild_role_rows(self) -> None:
        for child in self.role_frame.winfo_children():
            child.destroy()
        self._role_rows.clear()
        voices = self._system_voices()
        for i, rv in enumerate(self.settings.roles):
            name_var = tk.StringVar(value=rv.role)
            voice_var = tk.StringVar(value=rv.voice)
            lang_var = tk.StringVar(value=rv.language_hint or "")
            custom_var = tk.BooleanVar(value=rv.custom)
            direction_var = tk.StringVar(value=rv.direction)
            rate_var = tk.StringVar(value=rv.rate)
            pitch_var = tk.StringVar(value=rv.pitch)
            volume_var = tk.StringVar(value=rv.volume)

            ttk.Entry(self.role_frame, textvariable=name_var, width=14).grid(
                row=i, column=0, padx=theme.PAD_XS, sticky="w")
            cmb = ttk.Combobox(self.role_frame, textvariable=voice_var, width=26,
                               values=voices)
            cmb.grid(row=i, column=1, padx=theme.PAD_XS, sticky="w")
            ttk.Combobox(self.role_frame, textvariable=lang_var, width=6,
                         values=LANGUAGE_HINTS).grid(
                             row=i, column=2, padx=theme.PAD_XS)
            ttk.Checkbutton(self.role_frame, variable=custom_var).grid(
                row=i, column=3)
            ttk.Entry(self.role_frame, textvariable=direction_var, width=18).grid(
                row=i, column=4, padx=theme.PAD_XS, sticky="w")
            ent_rrate = ttk.Entry(self.role_frame, textvariable=rate_var, width=6)
            ent_rrate.grid(row=i, column=5, padx=theme.PAD_XS)
            ent_rpitch = ttk.Entry(self.role_frame, textvariable=pitch_var, width=6)
            ent_rpitch.grid(row=i, column=6, padx=theme.PAD_XS)
            ent_rvol = ttk.Entry(self.role_frame, textvariable=volume_var, width=6)
            ent_rvol.grid(row=i, column=7, padx=theme.PAD_XS)
            lbl_status = ttk.Label(self.role_frame, text="", width=32)
            lbl_status.grid(row=i, column=8, padx=theme.PAD_XS, sticky="w")
            ttk.Button(
                self.role_frame, text="试听", width=5,
                command=lambda rid=rv.id: self._on_audition_role(rid)
            ).grid(row=i, column=9)
            ttk.Button(
                self.role_frame, text="删除", width=5,
                command=lambda rid=rv.id: self._on_remove_role(rid)
            ).grid(row=i, column=10, padx=theme.PAD_XS)

            for var in (name_var, voice_var, lang_var, direction_var,
                        rate_var, pitch_var, volume_var):
                var.trace_add("write", lambda *_: self._on_role_row_changed())
            custom_var.trace_add("write", lambda *_: self._on_role_row_changed())

            self._role_rows[rv.id] = {
                "name": name_var, "voice": voice_var, "voice_widget": cmb,
                "lang": lang_var, "custom": custom_var,
                "direction": direction_var,
                "rate": rate_var, "pitch": pitch_var, "volume": volume_var,
                "rate_widget": ent_rrate, "pitch_widget": ent_rpitch,
                "volume_widget": ent_rvol,
                "status": lbl_status,
            }
        self._refresh_voice_choices()
        self._refresh_default_role_values()
        self._refresh_counts()
        self._sync_role_param_states()

    def _refresh_default_role_values(self) -> None:
        names = [rv.role for rv in self.settings.roles]
        self.cmb_default_role["values"] = [""] + names
        if self.var_default_role.get() not in names:
            self.var_default_role.set(
                self.settings.default_role
                if self.settings.default_role in names else "")

    def _refresh_counts(self) -> None:
        counts = self.settings.counts()
        known = (f"，系统音色 {counts['system_voices']}"
                 if counts["model_known"] else "，系统音色数未核实")
        self.lbl_counts.config(
            text=f"已配置角色 {counts['roles']} 个 · 已分配不同音色 "
                 f"{counts['distinct_voices']} 个{known}（并发请求数另由 "
                 "config.yaml 的 max_concurrency 控制，与角色数量无关）")
        if not self.var_distinct.get() and counts["roles"] >= 1:
            self.lbl_reuse.config(text="多个角色将使用同一个音色")
        else:
            self.lbl_reuse.config(text="")

    def _on_role_row_changed(self) -> None:
        # 改名/改音色/改 custom：先收集，再刷新状态与统计
        self._collect_from_rows()
        self._refresh_voice_choices()
        self._refresh_default_role_values()
        self._refresh_counts()
        self._schedule_save()

    def _on_distinct_toggle(self) -> None:
        self.settings.distinct_voices = bool(self.var_distinct.get())
        self._refresh_counts()
        self._schedule_save()

    def _on_add_role(self) -> None:
        self._collect_from_rows()
        name = simpledialog.askstring(
            "添加角色", "角色名称（可随时在表格中修改）：", parent=self)
        if name is None or not name.strip():
            return
        custom = messagebox.askyesno(
            "添加角色",
            "该角色是否使用自定义音色 ID（声音复刻/声音设计，custom）？\n\n"
            "“是”=不占用系统音色名额（云端 ID 是否存在无法离线验证）；\n"
            "“否”=从当前模型的系统音色中选择。",
            parent=self)
        try:
            if not custom:
                ok, reason = self.settings.can_add_non_custom_role()
                if not ok:
                    messagebox.showerror("无法添加角色", reason, parent=self)
                    return
            rv = self.settings.add_role(name.strip(), custom=bool(custom))
        except ConfigError as exc:
            messagebox.showerror("无法添加角色", str(exc), parent=self)
            return
        self.var_distinct.set(self.settings.distinct_voices)
        self._rebuild_role_rows()
        # 新建角色音色留空，明确等待用户选择（不自动填第一个）
        row = self._role_rows[rv.id]
        text, color = _STATUS_STYLE["empty"]
        row["status"].config(text=text, foreground=color)
        self._refresh_counts()
        self._schedule_save()

    def _on_remove_role(self, role_id: str) -> None:
        rv = self.settings.role_by_id(role_id)
        if rv is None:
            return
        self._collect_from_rows()
        try:
            self.settings.remove_role(role_id)
        except ConfigError as exc:
            messagebox.showerror("无法删除角色", str(exc), parent=self)
            return
        self._rebuild_role_rows()
        self._schedule_save()

    # —— 设置 ↔ 控件 ————————————————————————————————————————————————————

    def _sync_settings_to_widgets(self) -> None:
        s = self.settings
        self.var_key.set(s.api_key)
        self.var_provider.set(s.provider)
        self.var_model.set(s.model)
        self.var_distinct.set(s.distinct_voices)
        self._rebuild_role_rows()
        self._on_model_change()
        self.var_rate.set(s.rate)
        self.var_pitch.set(s.pitch)
        self.var_volume.set(s.volume)
        self.var_seed.set(s.seed)
        self.var_sample_rate.set(str(s.sample_rate))
        self.var_format.set(s.format)
        self.var_direction.set(s.direction)
        self.var_default_role.set(s.default_role)
        self.var_ssml.set(s.ssml_enabled)
        self.var_episode.set(s.episode_name)
        self.var_outdir.set(s.projects_dir)
        self._refresh_default_role_values()

    def _collect_from_rows(self) -> None:
        """把角色行当前值写回 self.settings.roles（按稳定 id 匹配）。"""
        for rv in self.settings.roles:
            row = self._role_rows.get(rv.id)
            if row is None:
                continue
            rv.role = row["name"].get().strip()
            rv.voice = row["voice"].get().strip()
            hint = row["lang"].get()
            rv.language_hint = hint or None
            rv.language_type = rv.language_type if not hint else None
            rv.custom = bool(row["custom"].get())
            rv.direction = row["direction"].get()
            rv.rate = row["rate"].get()
            rv.pitch = row["pitch"].get()
            rv.volume = row["volume"].get()

    def collect_settings(self) -> GuiSettings:
        self._collect_from_rows()
        roles = [
            RoleVoice(
                role=rv.role, voice=rv.voice,
                language_hint=rv.language_hint, language_type=rv.language_type,
                ssml=rv.ssml, custom=rv.custom, direction=rv.direction,
                rate=rv.rate, pitch=rv.pitch, volume=rv.volume, id=rv.id,
            )
            for rv in self.settings.roles
        ]
        s = GuiSettings(
            api_key=self.var_key.get().strip(),
            provider=self.var_provider.get(),
            model=self.var_model.get(),
            roles=roles,
            distinct_voices=bool(self.var_distinct.get()),
            rate=self.var_rate.get().strip(),
            pitch=self.var_pitch.get().strip(),
            volume=self.var_volume.get().strip(),
            seed=self.var_seed.get().strip(),
            sample_rate=int(self.var_sample_rate.get()),
            format=self.var_format.get(),
            direction=self.var_direction.get(),
            default_role=self.var_default_role.get(),
            ssml_enabled=bool(self.var_ssml.get()),
            episode_name=self.var_episode.get().strip() or "episode",
            projects_dir=self.var_outdir.get().strip(),
        )
        self.settings = s
        return s

    # —— 自动保存 ——————————————————————————————————————————————————————

    def _schedule_save(self) -> None:
        if self._save_after_id is not None:
            try:
                self.after_cancel(self._save_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._save_after_id = self.after(800, self._autosave_tick)

    def _autosave_tick(self) -> None:
        self._save_after_id = None
        self._save_if_valid(allow_missing_key=True, quiet=True)

    def _save_if_valid(self, *, allow_missing_key: bool, quiet: bool = False) -> bool:
        """配置（除 API Key 缺失外）无错误才落盘。返回是否已保存。"""
        s = self.collect_settings()
        errors = s.validate()
        if allow_missing_key:
            errors = [e for e in errors if "API Key" not in e]
        if errors:
            if not quiet:
                self._log("未自动保存（配置存在问题）：" + "；".join(errors[:3]))
            return False
        try:
            # 显式传路径：保证与界面显示/测试隔离使用同一个位置
            s.save(default_settings_path())
        except OSError as exc:  # noqa: BLE001
            self._log(f"自动保存失败：{exc}")
            return False
        if not quiet:
            masked = secret_store.mask_key(s.api_key)
            self._log(f"配置已保存：{default_settings_path()}"
                      + (f"（API Key {masked}，DPAPI 加密）" if masked else ""))
        return True

    # —— 动作 ————————————————————————————————————————————————————————

    def _guard_settings(self) -> GuiSettings | None:
        s = self.collect_settings()
        for w in s.warnings():
            self._log("提示：" + w)
        errors = s.validate()
        if errors:
            messagebox.showerror("设置校验未通过", "\n".join(errors), parent=self)
            return None
        return s

    def _on_start(self) -> None:
        s = self._guard_settings()
        if s is None:
            return
        # 合法配置（含合法 API Key）点击开始前自动保存
        if self._save_if_valid(allow_missing_key=False):
            masked = secret_store.mask_key(s.api_key)
            self._log(f"配置已保存（API Key {masked}，DPAPI 加密；日志不会显示完整 Key）。")
        try:
            ws = self._current_ws(s)
            # 无角色头且未选默认角色时，在启动前就明确提示（不静默使用任何角色）
            rows, errors = preview_working(s, ws)
        except ConfigError as exc:
            messagebox.showerror("无法开始", str(exc), parent=self)
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("无法生成预览", f"{type(exc).__name__}: {exc}",
                                 parent=self)
            return
        self._show_preview(s, ws, rows, errors)

    def _show_preview(self, s: GuiSettings, ws: WorkingScript,
                      rows: list[dict], errors: list[str]) -> None:
        """生成预览对话框（纯本地规划结果）；确认后才真正开始生成。"""
        win = tk.Toplevel(self)
        win.title("生成预览（确认后才会调用 API）")
        win.geometry("1260x600")
        win.transient(self)

        summary = preview_summary(rows)
        top = ttk.Frame(win)
        top.pack(fill="x", padx=8, pady=(8, 4))
        summary_txt = (
            f"共 {len(rows)} 段（台词 {summary['generate'] + summary['retry'] + summary['skip']} / "
            f"停顿 {summary['pause']}）：本次将调用 API {summary['generate'] + summary['retry']} 段"
            f"（新生成 {summary['generate'] - summary['dirty']}、"
            f"待更新重做 {summary['dirty']}、失败重试 {summary['retry']}），"
            f"跳过已最新 {summary['skip']} 段，停顿不调用 TTS。"
        )
        ttk.Label(
            top, text=summary_txt, foreground=theme.STATE_OK, wraplength=1220,
            justify="left", font=theme.FONT_BOLD,
        ).pack(anchor="w")
        ttk.Label(
            top,
            text="每段只发送其最终生效的一条 instruction；参数按 项目默认 < 角色默认 < 行内 "
                 "合并。长文本已在解析时安全切分。",
            foreground=theme.MUTED_FG, wraplength=1220, justify="left",
        ).pack(anchor="w", pady=(2, 0))
        if summary["generate"] + summary["retry"] == 0 and not errors:
            ttk.Label(
                top,
                text="所有片段音频都已是最新：本次不会调用云 API，仅做检查并在需要时"
                     "重新合并整章。",
                foreground=theme.STATE_PAUSE, wraplength=1220, justify="left",
            ).pack(anchor="w", pady=(2, 0))
        if errors:
            ttk.Label(
                top,
                text="存在必须先修正的问题，已禁止开始生成：\n" + "\n".join(errors),
                foreground=theme.STATE_ERROR, wraplength=1220, justify="left",
            ).pack(anchor="w", pady=(4, 0))

        table_frame = ttk.Frame(win)
        table_frame.pack(fill="both", expand=True, padx=8, pady=4)
        cols = ("idx", "kind", "state", "action", "role", "original", "text",
                "tags", "instruction", "params", "output", "error")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        headings = {
            "idx": ("序号", 50), "kind": ("类型", 55),
            "state": ("当前状态", 75), "action": ("本次动作", 110),
            "role": ("角色（显示名/音色外）", 120),
            "original": ("原文", 190), "text": ("实际发送文本", 220),
            "tags": ("情绪/拟声标签", 100),
            "instruction": ("最终 instruction", 160),
            "params": ("rate/pitch/volume", 110),
            "output": ("输出文件", 150), "error": ("问题", 180),
        }
        for key, (text, width) in headings.items():
            tree.heading(key, text=text)
            tree.column(key, width=width, anchor="w", stretch=(key == "text"))
        tree.tag_configure("pause", background=theme.ROW_PAUSE)
        tree.tag_configure("bad", background=theme.ROW_FAILED)
        tree.tag_configure("dirty", background=theme.ROW_DIRTY)
        tree.tag_configure("missing", background=theme.ROW_MISSING)
        tree.tag_configure("failed", background=theme.ROW_FAILED)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        bad = 0
        for r in rows:
            if r["kind"] == "pause":
                params = ""
                role = "（停顿）"
                kind_label = "停顿"
                tags = ("pause",)
            else:
                params = "/".join(
                    str(r[k]) if r[k] is not None else "—"
                    for k in ("rate", "pitch", "volume"))
                role = r["speaker_name"]
                # 类型：脚本对白 / 章节对白 / 章节普通正文（默认角色朗读）
                kind_label = {
                    "prose": "正文", "dialogue": "对白",
                }.get(r.get("origin"), "对白")
                tags = ("bad",) if r.get("error") else (r.get("state"),)
                if r.get("error"):
                    bad += 1
            # 自动分段补入的情绪标签与原文标签区分显示（⟦补入 …⟧ 前缀为系统自动加入）
            text_disp = r.get("text") or ""
            injected = r.get("injected")
            if injected and text_disp.startswith(injected):
                text_disp = (
                    f"⟦补入 {injected.strip()}⟧ " + text_disp[len(injected):]
                )
            tree.insert("", "end", values=(
                f"{r['index']:04d}", kind_label,
                r.get("state_zh", ""), r.get("action_zh", ""),
                role,
                r.get("original") or "", text_disp,
                r.get("tags") or "",
                r.get("instruction") or "（无）", params,
                r.get("output") or "", r.get("error") or "",
            ), tags=tags)

        bottom = ttk.Frame(win)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        def _do_start():
            win.destroy()
            try:
                self.controller.start_working(s, ws)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("无法启动", str(exc), parent=self)

        no_api = summary["generate"] + summary["retry"] == 0 and not errors and not bad
        btn_text = ("开始（所有片段已最新，不调用 API；仅检查/合并）"
                    if no_api else "确认无误，开始生成")
        btn_ok = ttk.Button(bottom, text=btn_text, command=_do_start)
        btn_ok.pack(side="right")
        if errors or bad:
            btn_ok.config(state="disabled")
        ttk.Button(bottom, text="取消", command=win.destroy).pack(
            side="right", padx=6)
        win.grab_set()

    def _on_audition_role(self, role_id: str) -> None:
        """角色页行内试听：临时把该行角色设为默认角色，复用统一试听弹窗。"""
        self._collect_from_rows()
        rv = self.settings.role_by_id(role_id)
        if rv is None:
            return
        base = self.collect_settings()
        if not base.api_key.strip():
            messagebox.showwarning(
                "无法试听", "请先在「设置」页填写 API Key。", parent=self)
            return
        # controller.audition 依赖 settings.default_role 定位音色；
        # 构造临时副本，不改动用户已保存的默认角色
        tmp = replace(base, default_role=rv.role)
        self._show_audition_dialog(tmp, rv.role)

    def _on_audition(self) -> None:
        """音色试听：只发一个短句，不写 manifest、不参与合并，目录只留最新一份。"""
        s = self._guard_settings()
        if s is None:
            return
        role = (s.default_role or "").strip()
        if not role:
            messagebox.showwarning(
                "无法试听",
                "请先在「设置」页的「默认角色（可选）」中选择要试听的角色，"
                "或到「角色」页点击该行角色的「试听」。",
                parent=self)
            return
        self._show_audition_dialog(s, role)

    def _show_audition_dialog(self, s: GuiSettings, role: str) -> None:
        dlg = tk.Toplevel(self)
        dlg.title(f"音色试听 — {role}")
        dlg.geometry("520x200")
        dlg.transient(self)
        frm = ttk.Frame(dlg)
        frm.pack(fill="both", expand=True, padx=12, pady=12)
        ttk.Label(
            frm,
            text="输入一句短试听文本（建议 30 字以内；试听只合成这一句，\n"
                 "不会写入项目、不影响任何片段，也不会拼接整章）：",
            justify="left",
        ).pack(anchor="w")
        var_text = tk.StringVar(value="你好，这是一段音色试听。")
        ttk.Entry(frm, textvariable=var_text).pack(fill="x", pady=8)

        def _do():
            sample = var_text.get().strip()
            if not sample:
                messagebox.showwarning("音色试听", "请输入试听文本。", parent=dlg)
                return
            if len(sample) > 120:
                messagebox.showwarning(
                    "音色试听", "试听仅用于确认音色，请输入 120 字以内的短句。",
                    parent=dlg)
                return
            dlg.destroy()
            try:
                self.controller.audition(s, sample)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("无法试听", str(exc), parent=self)

        bar = ttk.Frame(frm)
        bar.pack(fill="x", side="bottom")
        ttk.Button(bar, text="开始试听", command=_do).pack(side="right")
        ttk.Button(bar, text="取消", command=dlg.destroy).pack(side="right", padx=6)
        dlg.grab_set()

    def _on_retry_failed(self) -> None:
        """高级入口：用当前工作脚本把 未生成/失败/待更新 片段全部补齐（clean 跳过）。"""
        s = self._guard_settings()
        if s is None:
            return
        try:
            ws = self._current_ws(s)
            self.controller.start_working(s, ws)
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("无法生成未完成片段", str(exc), parent=self)

    def _on_regenerate(self) -> None:
        raw = simpledialog.askstring(
            "局部重生成（高级）", "输入要重做的片段编号（多个用逗号/空格分隔）：",
            parent=self)
        if not raw:
            return
        try:
            indices = [int(x) for x in raw.replace(",", " ").split()]
        except ValueError:
            messagebox.showwarning("局部重生成", "编号必须是整数。", parent=self)
            return
        try:
            self.controller.regenerate(indices)
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("无法局部重生成", str(exc), parent=self)

    def _on_merge_only(self) -> None:
        """高级入口：明确告知后用现有片段文件强制重拼（可能含待更新片段的旧音频）。"""
        ok = messagebox.askyesno(
            "仅重新合并（高级）",
            "将直接使用 segments 目录中的现有片段文件重新拼接整章，\n"
            "不会调用云 API。\n\n"
            "如果存在「待更新」片段，拼出的成品会包含它们的旧音频，\n"
            "不代表最新脚本效果。确定继续吗？",
            icon="warning", parent=self)
        if not ok:
            return
        try:
            self.controller.merge_only()
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("无法重新合并", str(exc), parent=self)

    # —— 片段管理（播放 / 编辑 / 单段重生成） ——————————————————————————————

    _SEG_STATE_ZH = {
        "pause": "停顿", "missing": "未生成", "failed": "生成失败",
        "dirty": "待更新", "clean": "已生成", "generating": "生成中",
    }

    def _on_manage_segments(self) -> None:
        s = self.collect_settings()
        try:
            ws = self._current_ws(s)
            rows = self.controller.segment_rows(s, ws)
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("无法打开片段管理", str(exc), parent=self)
            return
        win = tk.Toplevel(self)
        win.title(f"片段管理 — {ws.episode}")
        win.geometry("980x560")
        win.transient(self)

        top = ttk.Frame(win)
        top.pack(fill="x", padx=8, pady=(8, 4))
        src = ws.source_name or ("直接粘贴" if ws.source_type == "paste" else "旧项目")
        ttk.Label(
            top,
            text=f"来源：{src}　|　只生成 未生成/失败/待更新 的片段；"
                 "单段重生成只调用该段 TTS，其他音频不受影响。",
            foreground=theme.MUTED_FG, wraplength=940, justify="left",
        ).pack(anchor="w")

        # 状态筛选器（只影响显示，不影响数据）
        filter_bar = ttk.Frame(win)
        filter_bar.pack(fill="x", padx=8)
        ttk.Label(filter_bar, text="筛选状态：").pack(side="left")
        var_filter = tk.StringVar(value="全部")
        FILTER_CHOICES = ["全部", "已生成", "待更新", "未生成", "失败", "停顿"]
        cmb_filter = ttk.Combobox(
            filter_bar, textvariable=var_filter, state="readonly", width=10,
            values=FILTER_CHOICES)
        cmb_filter.pack(side="left", padx=(2, 8))
        FILTER_MAP = {"已生成": "clean", "待更新": "dirty", "未生成": "missing",
                      "失败": "failed", "停顿": "pause"}

        table = ttk.Frame(win)
        table.pack(fill="both", expand=True, padx=8, pady=4)
        cols = ("idx", "state", "role", "content", "params", "file")
        tree = ttk.Treeview(table, columns=cols, show="headings", selectmode="browse")
        headings = {
            "idx": ("序号", 60), "state": ("状态", 150), "role": ("角色", 110),
            "content": ("台词 / 停顿", 320), "params": ("rate/pitch/volume", 120),
            "file": ("音频文件", 180),
        }
        for key, (text, width) in headings.items():
            tree.heading(key, text=text)
            tree.column(key, width=width, anchor="w",
                        stretch=(key == "content"))
        tree.tag_configure("dirty", background=theme.ROW_DIRTY)
        tree.tag_configure("failed", background=theme.ROW_FAILED)
        tree.tag_configure("missing", background=theme.ROW_MISSING)
        tree.tag_configure("pause", background=theme.ROW_PAUSE)
        vsb = ttk.Scrollbar(table, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)

        # 持久选中：用稳定 seg_id，不依赖 Treeview 瞬时 selection
        selected_seg_id: list[str | None] = [None]

        def _row_by_seg_id(seg_id: str | None):
            if seg_id is None:
                return None
            for r in rows:
                if r.get("seg_id") == seg_id:
                    return r
            return None

        def _on_select(_evt=None):
            sel = tree.selection()
            if sel:
                # Treeview iid 就是 seg_id
                selected_seg_id[0] = sel[0]
            _update_buttons()

        def _update_buttons():
            has = selected_seg_id[0] is not None and _row_by_seg_id(selected_seg_id[0]) is not None
            for b in (btn_play, btn_edit, btn_regen):
                b.state(["!disabled"] if has else ["disabled"])

        def _reload():
            try:
                new_rows = self.controller.segment_rows(s, ws)
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning("刷新失败", str(exc), parent=win)
                return
            rows[:] = new_rows
            fkey = var_filter.get()
            want_state = FILTER_MAP.get(fkey)
            # 若当前选中的 seg_id 已不在新列表（重新导入/删除），清空选择
            if selected_seg_id[0] is not None and _row_by_seg_id(selected_seg_id[0]) is None:
                selected_seg_id[0] = None
            tree.delete(*tree.get_children())
            for r in rows:
                if want_state is not None and r["state"] != want_state:
                    continue
                if r["kind"] == "pause":
                    content = f"[PAUSE:{r['pause_seconds']:g}]"
                    params, role = "", "（停顿）"
                else:
                    content = r.get("text") or ""
                    role = r.get("speaker_name") or ""
                    params = "/".join(
                        str(r.get(k)) if r.get(k) is not None else "—"
                        for k in ("rate", "pitch", "volume"))
                state = self._SEG_STATE_ZH.get(r["state"], r["state"])
                if r.get("error"):
                    state += f"（{r['error']}）"
                # iid = seg_id，便于刷新后按稳定 ID 恢复选中
                tree.insert(
                    "", "end", iid=str(r["seg_id"]),
                    values=(
                        f"{r['index']:04d}", state, role, content, params,
                        r.get("file") or "",
                    ),
                    tags=(r["state"],),
                )
            # 恢复选中（仅当该 seg_id 仍存在且通过筛选）
            sid = selected_seg_id[0]
            if sid is not None and tree.exists(sid):
                tree.selection_set(sid)
                tree.see(sid)
            _update_buttons()

        def _selected_row():
            return _row_by_seg_id(selected_seg_id[0])

        def _play():
            row = _selected_row()
            if row is None:
                return
            if row["kind"] == "pause":
                messagebox.showinfo("停顿片段", "停顿不生成独立语音；它在最终合并时插入静音。",
                                    parent=win)
                return
            file_rel = row.get("file")
            cfg = s.to_app_config()
            path = cfg.projects_dir / ws.episode / file_rel if file_rel else None
            if not path or not path.is_file():
                messagebox.showinfo("尚无音频", "该片段还没有可用音频，请先生成。",
                                    parent=win)
                return
            if row.get("state") == "dirty":
                play_old = messagebox.askyesno(
                    "该片段已更新，现有音频是旧版本",
                    f"片段 {row['index']:04d} 的文本或参数已修改，磁盘上的音频"
                    "还是修改前合成的旧版本。\n\n是否仍要播放旧音频？\n"
                    "（点「生成 / 更新音频」后即可获得新版本。）",
                    icon="warning", parent=win)
                if not play_old:
                    return
            try:
                os.startfile(str(path))  # type: ignore[attr-defined]
            except OSError as exc:
                messagebox.showerror("无法播放", str(exc), parent=win)

        def _edit():
            row = _selected_row()
            if row is None:
                return
            self._edit_segment_dialog(win, s, ws, row, _reload)

        def _regen():
            row = _selected_row()
            if row is None:
                return
            if row["kind"] != "speech":
                messagebox.showinfo(
                    "停顿无需 TTS", "修改停顿后点「生成 / 更新音频」即可重新合并生效。",
                    parent=win)
                return
            if row.get("state") == "clean":
                ok = messagebox.askyesno(
                    "该片段已是最新",
                    f"片段 {row['index']:04d} 的音频与当前设置一致，重新生成会"
                    "再次调用云 API 并产生费用。\n\n确定要重新生成吗？",
                    icon="question", parent=win)
                if not ok:
                    return
            guard = self._guard_settings()
            if guard is None:
                return
            try:
                self.controller.regenerate_segments(guard, [row["seg_id"]], ws)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("无法重生成", str(exc), parent=win)
            finally:
                _reload()

        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        btn_play = ttk.Button(bar, text="播放", command=_play)
        btn_play.pack(side="left")
        btn_edit = ttk.Button(bar, text="编辑", command=_edit)
        btn_edit.pack(side="left", padx=4)
        btn_regen = ttk.Button(bar, text="重新生成此片段", command=_regen)
        btn_regen.pack(side="left")
        ttk.Button(bar, text="刷新", command=_reload).pack(side="left", padx=8)
        ttk.Button(bar, text="关闭",
                   command=lambda: _on_close()).pack(side="right")
        # 底部状态栏：显示当前选中项
        status_bar = ttk.Label(bar, text="", foreground=theme.MUTED_FG)
        status_bar.pack(side="right", padx=(0, 8))

        def _update_status():
            row = _selected_row()
            if row is None:
                status_bar.config(text="")
            else:
                state = self._SEG_STATE_ZH.get(row.get("state", ""), row.get("state", ""))
                status_bar.config(
                    text=f"# {row.get('index', '?'):04d}  {state}  "
                         f"{row.get('speaker_name', '') or '（停顿）'}")

        orig_update = _update_buttons
        def _update_buttons():
            orig_update()
            _update_status()
        _update_buttons()

        cmb_filter.bind("<<ComboboxSelected>>", lambda _e: _reload())
        tree.bind("<<TreeviewSelect>>", _on_select)
        tree.bind("<Double-1>", lambda _e: _edit())  # 双击直接编辑

        _reload()
        win.grab_set()

        def _on_close():
            win.destroy()
            # 关闭片段管理后刷新主界面状态摘要（编辑/重生成可能已改变状态）
            self._refresh_summary()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        # 生成任务在后台跑时，窗口保持打开展示最新状态
        def _tick():
            if win.winfo_exists():
                if not self.controller.busy:
                    _reload()
                    self._refresh_summary()
                win.after(1500, _tick)
        win.after(1500, _tick)

    def _edit_segment_dialog(self, parent, s, ws: WorkingScript, row: dict,
                             on_saved) -> None:
        """单段编辑弹窗：speech 改 正文/角色/导演指令/参数；pause 改时长。

        保存写入工作脚本覆盖层（seg_id 为键）并立即落盘；仅该段随后变 dirty。
        """
        dlg = tk.Toplevel(parent)
        dlg.title(f"编辑片段 {row['index']:04d}")
        dlg.geometry("560x460")
        dlg.transient(parent)
        frm = ttk.Frame(dlg)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        if row["kind"] == "pause":
            ttk.Label(frm, text="停顿时长（秒，0-10；不调用 TTS）").pack(anchor="w")
            var_dur = tk.StringVar(value=f"{row['pause_seconds']:g}")
            ttk.Entry(frm, textvariable=var_dur).pack(fill="x", pady=2)

            def _save_pause():
                try:
                    val = float(var_dur.get().strip())
                except ValueError:
                    messagebox.showwarning("编辑片段", "时长必须是数字。", parent=dlg)
                    return
                self.controller.edit_segment(
                    s, row["seg_id"], {"pause_seconds": val}, ws=ws)
                dlg.destroy()
                on_saved()

            ttk.Button(frm, text="保存", command=_save_pause).pack(anchor="e", pady=8)
            return

        ttk.Label(frm, text="正文").pack(anchor="w")
        txt = tk.Text(frm, height=6, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", row.get("text") or "")

        role_line = ttk.Frame(frm)
        role_line.pack(fill="x", pady=(6, 2))
        ttk.Label(role_line, text="角色").pack(side="left")
        names = [rv.role for rv in s.roles]
        ids = [rv.id for rv in s.roles]
        cur_name = row.get("speaker_name") or ""
        var_role = tk.StringVar(
            value=cur_name if cur_name in names else "")
        ttk.Combobox(role_line, textvariable=var_role, values=names,
                     state="readonly", width=24).pack(side="left", padx=4)

        ttk.Label(frm, text="局部导演/表演指令（留空=清除该片段局部指令，回退到角色/项目级）").pack(
            anchor="w", pady=(4, 0))
        var_dir = tk.StringVar(value=row.get("direction") or "")
        ttk.Entry(frm, textvariable=var_dir).pack(fill="x")

        # rate/pitch/volume 预填"覆盖层"值（无覆盖则留空），避免把默认值冻结成覆盖
        ov = ws.overrides.get(row["seg_id"], {}) if ws else {}
        param_line = ttk.Frame(frm)
        param_line.pack(fill="x", pady=6)
        ttk.Label(param_line, text="rate").pack(side="left")
        var_rate = tk.StringVar(value="" if ov.get("rate") is None else str(ov["rate"]))
        ttk.Entry(param_line, textvariable=var_rate, width=7).pack(
            side="left", padx=(2, 10))
        ttk.Label(param_line, text="pitch").pack(side="left")
        var_pitch = tk.StringVar(value="" if ov.get("pitch") is None else str(ov["pitch"]))
        ttk.Entry(param_line, textvariable=var_pitch, width=7).pack(
            side="left", padx=(2, 10))
        ttk.Label(param_line, text="volume").pack(side="left")
        var_vol = tk.StringVar(value="" if ov.get("volume") is None else str(ov["volume"]))
        ttk.Entry(param_line, textvariable=var_vol, width=7).pack(
            side="left", padx=2)
        ttk.Label(param_line, text="（留空 = 回退默认）",
                  foreground=theme.MUTED_FG).pack(side="left", padx=6)

        def _save():
            new_text = txt.get("1.0", "end").strip("\n").strip()
            if not new_text:
                messagebox.showwarning("编辑片段", "正文不能为空。", parent=dlg)
                return
            patch = {"text": new_text}
            name = var_role.get().strip()
            if name:
                i = names.index(name)
                patch["speaker"] = ids[i]
            # direction 始终写入：空字符串 = 显式清除该片段局部指令（不恢复脚本里的行内指令）
            patch["direction"] = var_dir.get().strip()
            for key, var in (("rate", var_rate), ("pitch", var_pitch),
                             ("volume", var_vol)):
                val = var.get().strip()
                if val:
                    try:
                        patch[key] = float(val) if key != "volume" else int(float(val))
                    except ValueError:
                        messagebox.showwarning(
                            "编辑片段", f"{key} 必须是数字。", parent=dlg)
                        return
                else:
                    # 留空 = 清除该参数的片段级覆盖，回退到角色/项目默认
                    patch[key] = None
            try:
                self.controller.edit_segment(s, row["seg_id"], patch, ws=ws)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("保存失败", str(exc), parent=dlg)
                return
            dlg.destroy()
            on_saved()

        ttk.Button(frm, text="保存（仅该片段标记为待更新）",
                   command=_save).pack(anchor="e", pady=8)

    def _pick_outdir(self) -> None:
        d = filedialog.askdirectory(title="选择输出目录（projects）", parent=self)
        if d:
            self.var_outdir.set(d)

    def _on_text_modified(self, _event) -> None:
        """文本框手动改动追踪（程序填充时先解绑/复位标志）。"""
        if self.txt_input.edit_modified():
            self._text_dirty = True
            self.txt_input.edit_modified(False)
            # 用户真实编辑后，占位符标志必须清除（避免后续误把内容当占位清空）
            if self._placeholder_active and self.txt_input.get("1.0", "end").strip() != PLACEHOLDER_TEXT:
                self._placeholder_active = False

    def _fill_textbox(self, text: str) -> None:
        """程序填充文本框（导入文件）：不把这次填充记为用户改动。"""
        self.txt_input.delete("1.0", "end")
        self.txt_input.insert("1.0", text)
        self.txt_input.config(foreground=theme.FG)
        self._placeholder_active = False
        self.txt_input.edit_modified(False)
        self._text_dirty = False

    def _dry_ws(self, settings: GuiSettings) -> WorkingScript:
        """按界面当前内容构造工作脚本做只读检查；不落盘、不改变 self.ws。"""
        episode = settings.episode_name.strip() or "episode"
        text = self.txt_input.get("1.0", "end").strip("\n")
        if self._placeholder_active and text == PLACEHOLDER_TEXT:
            text = ""
        book = self.var_book_path.get()
        if book:
            return WorkingScript.import_file(episode, Path(book), mode="book")
        return WorkingScript.from_paste(episode, text)

    def _refresh_summary(self) -> None:
        """刷新主界面状态摘要（纯本地：不触网、不落盘、不改工作脚本状态）。"""
        try:
            s = self.collect_settings()
            ws = self._current_ws(s) if self.ws is not None else self._dry_ws(s)
            info = chapter_summary(s, ws)
        except Exception:  # noqa: BLE001 —— 摘要只是辅助显示，任何异常都静默
            self.lbl_summary.config(text="", foreground=theme.FG)
            return
        if "error" in info:
            self.lbl_summary.config(
                text="（完成配置并选择/粘贴脚本后，这里会显示片段与整章音频状态）",
                foreground=theme.MUTED_FG)
            return
        c = info["counts"]
        busy_tail = "　|　正在生成…" if self.controller.busy else ""
        self.lbl_summary.config(
            text=(
                f"共 {info['total']} 段（台词 {info['speech_total']}）："
                f"已生成 {c['clean']} / 待更新 {c['dirty']} / "
                f"未生成 {c['missing']} / 失败 {c['failed']} / 停顿 {c['pause']}"
                f"　|　整章音频：{info['state_zh']}{busy_tail}"
            ),
            foreground=theme.FG)

    def _on_play_chapter(self) -> None:
        """播放整章成品（纯本地，绝不调用 TTS/不触发合并）。"""
        try:
            s = self.collect_settings()
            cfg = s.to_app_config()
            episode = s.episode_name.strip() or "episode"
            path = (cfg.projects_dir / episode
                    / "output" / f"{episode}.{cfg.merge_format}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showinfo("无法播放整章", str(exc), parent=self)
            return
        if not path.is_file():
            messagebox.showinfo(
                "整章尚未生成",
                "还没有整章成品音频。请先点「生成 / 更新音频」，全部片段生成后"
                "会自动拼接为整章。",
                parent=self)
            return
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("无法播放", str(exc), parent=self)

    def _on_export_srt(self) -> None:
        """导出句级 SRT：时间轴由最终拼接序列同源计算（纯本地，不触网）。"""
        try:
            s = self.collect_settings()
            episode = s.episode_name.strip() or "episode"
            ws = self._current_ws(s)  # 确保守卫看到文本框的最新内容
            self.controller.episode = episode
            self.controller.set_settings(s)
            content, count = self.controller.build_srt()
        except Exception as exc:  # noqa: BLE001  配置/守卫/ffprobe 错误统一提示
            messagebox.showinfo("暂不能导出字幕", str(exc), parent=self)
            return
        if count == 0:
            messagebox.showinfo(
                "没有可导出的字幕",
                "当前项目没有台词片段（可能只含停顿），没有可生成的句级字幕。",
                parent=self)
            return
        target = filedialog.asksaveasfilename(
            title="导出 SRT 字幕",
            parent=self,
            initialdir=str(Path.home()),
            initialfile=f"{episode}.srt",
            defaultextension=".srt",
            filetypes=[("SRT 字幕", "*.srt"), ("所有文件", "*.*")])
        if not target:
            return
        try:
            Path(target).write_text(content, encoding="utf-8", newline="\n")
        except OSError as exc:
            messagebox.showerror("写出字幕失败", str(exc), parent=self)
            return
        messagebox.showinfo(
            "字幕已导出",
            f"已导出 {count} 条句级字幕：\n{target}\n\n"
            "时间轴与最终拼接的音频同源（含开头/中间/结尾停顿）。\n"
            "若之后重新生成过片段或调整停顿，请重新导出。",
            parent=self)

    def _current_ws(self, settings: GuiSettings) -> WorkingScript:
        """获取与界面同步的当前工作脚本（粘贴/文件统一）。

        - 已有 ws：同步 episode 与文本框内容（用户改动 → update_text 标 modified）；
        - 无 ws：按是否已导入文件构造 import_file / from_paste。
        """
        episode = settings.episode_name.strip() or "episode"
        text = self.txt_input.get("1.0", "end").strip("\n")
        if self._placeholder_active and text == PLACEHOLDER_TEXT:
            text = ""
        book = self.var_book_path.get()
        if self.ws is None:
            if book:
                ws = WorkingScript.import_file(episode, Path(book), mode="book")
            else:
                ws = WorkingScript.from_paste(episode, text)
            self.ws = ws
            self.controller.ws = ws
            return ws
        ws = self.ws
        ws.episode = episode
        ws.update_text(text + ("\n" if text else ""))
        return ws

    def _pick_book(self) -> None:
        p = filedialog.askopenfilename(
            title="选择要导入的文本/Markdown 文件", parent=self,
            filetypes=[("文本/Markdown", "*.md *.markdown *.txt"),
                       ("所有文件", "*.*")])
        if not p:
            return
        # 重新导入保护：当前工作脚本/文本框有未同步修改时必须显式确认
        if (self.ws is not None and self.ws.modified) or self._text_dirty:
            ok = messagebox.askyesno(
                "重新导入会覆盖当前工作脚本",
                "当前工作脚本包含未同步的修改，重新导入会替换这些内容，是否继续？\n\n"
                "（原始文件始终只读；被替换的是 Podvia 内部工作脚本。）",
                icon="warning", parent=self)
            if not ok:
                return
        try:
            ws = WorkingScript.import_file(Path(p).stem, Path(p), mode="book")
        except Exception as exc:  # noqa: BLE001 —— 读取/编码失败直接提示
            messagebox.showerror("导入失败", f"{type(exc).__name__}: {exc}",
                                 parent=self)
            return
        ws.episode = self.var_episode.get().strip() or Path(p).stem
        self.ws = ws
        self.controller.ws = ws
        self.var_book_path.set(p)
        self._fill_textbox(ws.text)
        self.lbl_book.config(
            text=f"来源：{Path(p).name} · 已导入到 Podvia 工作脚本",
            foreground=theme.STATE_OK)
        self.btn_clear_book.state(["!disabled"])
        if self.var_episode.get().strip() in ("", "episode"):
            self.var_episode.set(Path(p).stem)
        self._log(
            f"已导入文件：{p} → 内容已复制为工作脚本（原始文件保持只读，不会回写）。")
        self._refresh_summary()

    def _clear_book(self) -> None:
        """清除来源：文本框内容即新的粘贴型工作脚本（下次生成时生效）。"""
        self.var_book_path.set("")
        self.ws = None
        self.controller.ws = None
        self._text_dirty = True  # 框内现存文本视为用户内容，防止下次导入静默覆盖
        self.lbl_book.config(
            text="来源：直接粘贴（文本框内容将作为工作脚本）",
            foreground=theme.MUTED_FG)
        self.btn_clear_book.state(["disabled"])
        self._refresh_summary()

    def _insert_tag(self, tag: str) -> None:
        """在文本光标处插入官方情绪/拟声标签（不覆盖已有台词）。"""
        if self._placeholder_active:
            self._clear_placeholder()
        self.txt_input.insert("insert", tag + " ")
        self.txt_input.focus_set()

    # —— FFmpeg 启动提示（不运行 -version，只看文件/PATH） ——————————————————

    def _log_ffmpeg_hint(self) -> None:
        try:
            from podcast_tts.audio import resolve_tools
            from podcast_tts.config import load_config

            cfg = load_config(key_required=False)
            ffmpeg, ffprobe = resolve_tools(cfg)
        except Exception:  # noqa: BLE001 —— 仅提示，不影响启动
            return
        found = (Path(ffmpeg).is_file() or shutil.which(Path(ffmpeg).name)) and (
            Path(ffprobe).is_file() or shutil.which(Path(ffprobe).name))
        if not found:
            self._log(
                "未检测到 FFmpeg/ffprobe：片段可以生成，但「拼接」会失败。\n"
                "       解决办法：从 https://www.gyan.dev/ffmpeg/builds/ 下载 "
                "release-full，把 ffmpeg.exe、ffprobe.exe 放入 "
                "%APPDATA%\\Podvia\\tools\\bin，或将其 bin 目录加入 PATH。"
            )

    # —— 事件轮询（主线程） ——————————————————————————————————————————

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.controller.events.get_nowait()
            except Exception:  # noqa: BLE001
                break
            self._handle_event(event)
        self.after(120, self._poll_events)

    def _apply_state(self, state: str) -> None:
        busy = state in _BUSY_STATES
        self.btn_start.config(state="disabled" if busy else "normal")
        # 暂停/继续合并为单一按钮，running 时显示"暂停"，paused 时显示"继续"
        if state == "running":
            self.btn_pause.config(text="暂停", state="normal")
        elif state == "paused":
            self.btn_pause.config(text="继续", state="normal")
        else:
            self.btn_pause.config(text="暂停", state="disabled")
        self.btn_cancel.config(
            state="normal" if state in ("running", "paused") else "disabled")
        # 任务运行期间禁止发起新的规划/导出类操作
        for b in (self.btn_manage, self.btn_export, self.mbtn_advanced):
            b.state(["disabled"] if busy else ["!disabled"])
        self.lbl_progress.config(text=STATUS_TEXT.get(state, state))
        if busy:
            self.progress.config(mode="indeterminate")
            self.progress.start(15)
        else:
            self.progress.stop()
            self.progress.config(mode="determinate")
        self._refresh_summary()

    def _handle_event(self, event: tuple) -> None:
        kind = event[0]
        if kind == "status":
            self._apply_state(event[1])
        elif kind == "progress":
            _, idx, status, speaker, error = event
            self._log(f"片段 {idx:04d} {speaker} → {status}"
                      + (f"：{error}" if error else ""))
        elif kind == "log":
            self._log(event[1])
        elif kind == "failed":
            self._failed = event[1]
            for item in self._failed[:10]:
                self._log(f"失败 {item['index']:04d} {item['speaker']}：{item['error']}")
        elif kind == "fatal":
            self._log("[致命错误] " + event[1])
            messagebox.showerror("运行错误", event[1], parent=self)
        elif kind == "merged":
            info = event[1]
            self._log(f"拼接成功，成品输出：{info['output']}")
            if info.get("srt"):
                srt_abs = (Path(info["output"]).parent
                           / Path(info["srt"]).name)
                self._log(f"同步生成句级字幕（与成品同源）：{srt_abs}，"
                          "可用「导出字幕…」另存到任意位置。")
        elif kind == "merge_failed":
            info = event[1]
            self._failed_merge = info
            self._log(f"片段生成完成但拼接失败：{info['error']}")
            self._log(f"最终输出路径：{info['output']}（片段已保留）")
            messagebox.showerror(
                "片段生成完成但拼接失败",
                f"错误：{info['error']}\n\n最终输出路径：{info['output']}\n\n"
                "片段文件已全部保留，可在「高级 ▾」中点「仅重新合并」重试，"
                "不需要重新请求 API。",
                parent=self)
        elif kind == "merge_skipped":
            info = event[1]
            self._log(
                f"还有 {len(info.get('pending', []))} 个片段待更新，"
                "本次保留片段生成结果，未产出最新整章。")
        elif kind == "finished":
            self._on_finished(event[1])
        elif kind == "audition_done":
            self._log(f"音色试听已生成：{event[1]}（已自动用默认播放器打开）")
            try:
                os.startfile(event[1])  # type: ignore[attr-defined]
            except OSError:
                pass

    def _on_finished(self, info: dict) -> None:
        total = info.get("total", 0)
        done = info["succeeded"] + info["failed"]
        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = done
        self._log(
            f"结束：成功 {info['succeeded']} / 失败 {info['failed']} / "
            f"跳过 {info['skipped']}" + ("（已取消）" if info.get("cancelled") else ""))
        merged = info.get("merged")
        if merged is True:
            if info.get("output"):
                self._log(f"成品输出路径：{info['output']}")
            if info.get("merge_skipped"):
                self._log("片段未变化，沿用已通过 ffprobe 识别的成品文件。")
        elif info.get("merge_skipped"):
            self._log("仍有片段未更新：未产出最新整章（本次不会拼接旧音频）。"
                      "继续点「生成 / 更新音频」补齐其余片段后会自动拼接。")
        elif merged is False and not info.get("cancelled") and info["failed"] == 0:
            self._log("片段已全部生成但拼接未成功，可在「⋯」菜单中点「仅重新合并」"
                      "重试（不会重新调用 API）。")
        self._refresh_summary()

    def _log(self, msg: str) -> None:
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    def _on_close(self) -> None:
        if self.controller.busy:
            if not messagebox.askokcancel(
                "确认退出",
                "任务仍在运行。退出不会撤回已发出的云端请求；建议先取消并等待在途片段结束。",
                parent=self,
            ):
                return
        # 关闭窗口：保存除 Key 缺失外无错误的配置（Key 通过校验则随 DPAPI 一起保存）
        try:
            self._save_if_valid(allow_missing_key=True, quiet=True)
        except Exception:  # noqa: BLE001 —— 关闭不被保存异常阻断
            pass
        self.destroy()


def main() -> None:
    PodviaApp().mainloop()


if __name__ == "__main__":
    main()
