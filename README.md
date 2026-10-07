# Podvia

**Podvia 是一款 Windows 桌面多角色语音合成工具**：粘贴对话脚本或导入章节文件，按角色分配音色，逐段调用阿里云百炼（DashScope）Qwen-Audio-TTS 合成，再自动拼接成完整音频。适用于多人对话、播客节目、广播剧、有声书配音等场景。

界面基于 Python Tkinter，开箱即用；同时提供完整的命令行接口。

## 功能特性

* **两种输入方式**：直接粘贴对话脚本，或导入 `.md` / `.txt` 章节文件
* **多角色多音色**：每个角色绑定固定音色（保证音色一致性），支持中文角色名与 `\[旁白]` 标记
* **导演指令四级层级**：项目 < 角色 < 章节 < 行内（如 `Lily(轻快地)：…`），行内优先
* **语音参数三级覆盖**：`rate` / `pitch` / `volume` 支持项目级、角色级、片段级逐层覆盖
* **情绪与人声拟声标签**：官方 30 个标签（如 `\[excited]`、`\[gasp]`、`\[whispers]`），GUI 提供中文插入菜单
* **SSML 支持**：显式开启后按白名单校验（`speak` / `break` / `phoneme` / `sub` / `say-as`）
* **停顿行**：`\[PAUSE:1.0]` 独占一行，合并时插入对应时长静音，不消耗 API 调用
* **生成前逐段预览**：确认角色、指令、参数无误后再发起合成
* **片段级缓存**：内容指纹未变化的片段自动跳过，只重生成修改过的部分
* **片段管理器**：表格查看全部片段，可直接编辑正文/导演指令/参数、播放、单段重生成
* **失败处理**：失败自动重试（指数退避）、一键重试失败片段、指定编号局部重生成
* **自动安全分段**：长台词按句读切分（安全线 480 字），不切断数字/标签，情绪跨段自动延续
* **SRT 字幕导出**：按最终拼接时间轴生成句级字幕，与成品音频同源

## 界面预览

**创作页** —— 粘贴脚本 / 导入章节，插入情绪标签与停顿，发起生成、暂停与导出：

!\[创作页](docs/screenshots/create.png)

**角色页** —— 角色 ↔ 音色绑定，支持试听，可为每个角色使用不同音色：

!\[角色页](docs/screenshots/roles.png)

**设置页** —— API Key（DPAPI 加密保存）、模型选择与全局语音参数：

!\[设置页](docs/screenshots/settings.png)

## 工作原理

```
对话脚本 / 章节文件
        ↓  解析（角色头、导演指令、行内参数、停顿、情绪标签）
片段列表（每段 ≤480 字，超长自动安全切分）
        ↓  内容指纹比对（未变化的片段直接复用已有音频）
逐段调用 Qwen-Audio-TTS（音色 + 指令 + 语音参数）
        ↓  片段音频落盘
FFmpeg 拼接（格式统一重采样 → 插入停顿静音 → 一次编码）
        ↓
成品音频 projects/<episode>/output/<episode>.mp3（可另导出 SRT）
```

每个片段生成成功后记录内容指纹与参数快照；修改任意文本、指令或参数后，只有该片段会被重新合成。

## 快速开始

### 方式一：下载成品（推荐）

1. 到 [Releases](../../releases) 下载最新 ZIP；
2. 解压整个文件夹（`Podvia.exe` 依赖同目录的 `\_internal`，不可单独移动）；
3. 双击 `Podvia.exe`，按提示填入百炼 API Key（经 Windows DPAPI 加密保存在本机）。

### 方式二：从源码运行

需要 Python 3.10+（python.org Windows 安装器自带 Tkinter）：

```powershell
git clone https://github.com/Lanyue1209/Podvia.git
cd Podvia
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
.venv\\Scripts\\python.exe main.py        # 启动 GUI（不带参数时默认）
.venv\\Scripts\\python.exe -m pytest tests -q   # 运行测试（全部 mock，不触网）
```

### FFmpeg

合成音频与导出字幕依赖 FFmpeg / FFprobe（合并、重采样、静音插入均由它完成）：

* 将 `ffmpeg.exe`、`ffprobe.exe` 放入项目 `tools/bin/`（此目录不进版本库），或
* 安装到系统 PATH；

程序按 `tools/bin → 系统 PATH` 的顺序自动查找。

### API Key

* **GUI 用户**：首次启动在「设置」页填入，Key 由 DPAPI 加密保存于 `%APPDATA%\\Podvia\\gui\_settings.json`，不会明文落盘；
* **命令行用户**：在项目根目录创建 `.env` 文件写入 `DASHSCOPE\_API\_KEY=sk-xxx`（`.env` 已被 gitignore，不会提交）。

API Key 请在[阿里云百炼控制台](https://bailian.console.aliyun.com/)开通并创建。注意：使用 API 产生的费用由你的账号自行承担。

## 输入格式

对话脚本示例：


Lily(轻快地)：你终于到了！
Leo：抱歉抱歉，路上堵车。
\[PAUSE:0.5]
\[旁白] 两人一前一后走进了剧场。
Lily：\[gasp] 这就是传说中的那个地方吗？


* `角色名：台词` 即角色行；括号内为导演指令，可附加行内参数：`Lily(轻快地; rate=1.1, volume=80)：…`
* `\[PAUSE:秒数]` 独占一行插入停顿；`\[旁白] 台词` 使用已注册的旁白角色
* 情绪/拟声标签（`\[excited]`、`\[gasp]` 等）写在台词文本内，仅限 Qwen-Audio-TTS 三个模型
* SSML 需显式开启；同一片段内 SSML 与情绪标签不可混用

完整规则（角色注册、章节模式、标签清单、参数范围、常见错误）见 [**输入格式规范.md**](输入格式规范.md)。

## 命令行

不带参数运行 `main.py` 默认启动 GUI；所有功能也可纯命令行完成：

|命令|作用|
|-|-|
|`python main.py gui`|启动图形界面|
|`python main.py parse <script.md>`|解析脚本，打印片段摘要（调试用）|
|`python main.py generate <script> \[--force] \[--no-merge]`|解析 → 生成 → 自动合并|
|`python main.py book <file.md/.txt> \[--speaker 角色]`|导入章节文件，分段 → 生成 → 自动合并|
|`python main.py retry <episode>`|重试 failed/pending 片段，成功后自动合并|
|`python main.py regen <episode> <编号...>`|忽略缓存，重生成指定编号片段|
|`python main.py merge <episode> \[--force]`|仅重新合并已有片段音频|
|`python main.py srt <episode> \[--output 路径]`|按拼接时间轴导出 SRT 字幕|
|`python main.py status <episode>`|查看片段成功/失败/合并状态摘要|

## 模型与配置

* 默认模型 `qwen-audio-3.0-tts-plus`，亦支持 `qwen-audio-3.1-tts-flash`、`qwen-audio-3.0-tts-flash`（情绪/拟声标签仅这三个模型支持）；
* 默认角色 Lily / Leo 绑定 Qwen-Audio-TTS 系统音色 `longanlingxin`（女）与 `longanlufeng`（男），可在角色表中更换；
* 全部默认值与可选项见 [config.yaml](config.yaml)（含采样率、格式、并发、重试、SSML、hot\_fix 发音修正等注释说明）。

## 配置与数据存放

运行期所有可写数据都在用户目录，程序目录保持只读：

```
%APPDATA%\\Podvia\\
├─ gui\_settings.json        界面配置（API Key 经 DPAPI 加密）
├─ projects\\<episode>\\
│  ├─ working\_script.json   工作脚本 + 片段级修改
│  ├─ metadata\\manifest.json  片段账本（状态 / 内容指纹 / 参数快照）
│  ├─ segments\\\*.wav        单片段音频
│  └─ output\\\*.mp3          最终成品
└─ logs\\                    运行日志
```

## 从源码打包

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build.ps1
```

使用 PyInstaller 构建 `dist/Podvia/` 绿色版。注意：构建机会把本地 `tools/bin` 中的 FFmpeg 一并打入安装包；向外分发安装包时请遵守 FFmpeg 的开源许可证（附许可证文本与源码链接），详见 `packaging/使用说明.txt`。

## 已知限制

* 单次请求文本上限 600 字（API 限制），程序在 480 字处安全切分；
* 不支持背景音乐、环境音效与多轨混音；
* GUI 仅面向 Windows（依赖 Win32 DPAPI 与 Tkinter）；命令行部分同样依赖 Windows 路径约定；
* 真实合成效果与音质以百炼线上服务为准，单元测试全部使用 mock，不产生 API 费用。

## 免责声明

本项目与阿里巴巴/阿里云无任何隶属关系；使用第三方 API 产生的费用、数据与内容合规责任由使用者自行承担。请遵守所在地区法律法规及阿里云服务条款，勿用于合成侵犯他人权益的内容。

## 许可证

[MIT](LICENSE) © 2026 Lanyue1209

分发包含 FFmpeg 的安装包时，FFmpeg 适用其自身开源许可证（LGPL/GPL，取决于构建配置），需单独附带许可说明与源码链接。

