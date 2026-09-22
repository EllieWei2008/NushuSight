# NüshuSight — Handwritten Nüshu OCR Studio / 女书手写体识别工具

NüshuSight is a local OCR tool for recognizing handwritten **Nüshu** (女书) characters — the world's only known female-exclusive writing system. It uses a Siamese Network (VGG16 backbone) trained with one-shot mixed learning, supporting all 396 Unicode Nüshu characters (U+1B170–U+1B2FF).

**Accuracy on handwritten samples:** Top-1 ~70.8% · Top-5 ~95%

---

NüshuSight 是一款本地运行的女书手写体 OCR 识别工具，支持 Unicode 标准收录的全部 396 个女书字符（U+1B170–U+1B2FF）。识别模型基于孪生网络（VGG16主干）+ 一次性混合训练方法构建。**手写样本识别率：** Top-1 约 70.8% · Top-5 约 95%
<img width="604" height="806" alt="Screenshot 2026-09-22 at 13 25 46" src="https://github.com/user-attachments/assets/3c4d8e09-dd27-496f-bb55-95f045f26593" />
**手写样本识别率：** Top-1 约 70.8% · Top-5 约 95%

---

## Table of Contents / 目录

- [English Instructions](#english-instructions)
- [中文使用说明](#中文使用说明)

---

# English Instructions

## 1. Download the Code

### Option A — Download as ZIP (Recommended for most users)

You do not need to install Git for this option.

1. Open your browser and go to:
   **[https://github.com/EllieWei2008/NushuSight](https://github.com/EllieWei2008/NushuSight)**
2. Click the green **`Code`** button near the top-right of the page.
3. In the dropdown menu, click **`Download ZIP`**.
4. Save the ZIP file to your computer, for example to your Desktop.
5. Right-click the downloaded ZIP file and choose **Extract All** (Windows) or double-click to unzip (Mac).
6. You should now have a folder called `NushuSight-main`. You can rename it to `NushuSight` for convenience.

### Option B — Clone with Git (for users familiar with Git)

```bash
git clone https://github.com/EllieWei2008/NushuSight.git
```

---

## 2. Download the Model Weights

The recognition model file (`oneshot_best.pth`, approximately 106 MB) is too large to store directly in the code repository. It is provided separately and must be downloaded and placed in the correct location manually.

**Step 1 — Download the file**

1. Go to the Releases page:
   **[https://github.com/EllieWei2008/NushuSight/releases](https://github.com/EllieWei2008/NushuSight/releases)**
2. Click on the latest release (at the top of the list).
3. Under **Assets**, click **`oneshot_best.pth`** to download it.

**Step 2 — Place the file in the correct folder**

Inside your `NushuSight` folder, there is a subfolder called `checkpoints`, and inside it another folder called `oneshot`. Place the downloaded file there:

```
NushuSight/
└── checkpoints/
    └── oneshot/
        └── oneshot_best.pth   ← place the downloaded file here
```

> If the `checkpoints/oneshot/` folders do not exist yet, create them manually.

---

## 3. Set Up the Environment

NüshuSight runs on **Python**, which is a free programming language runtime. You do not need to write any code — Python is only needed to run the tool. You also need a compatible web browser to use the interface.

### 3.1 Check if Python is already installed

**On Windows:**
1. Press `Win + R`, type `cmd`, and press Enter to open the Command Prompt.
2. Type the following and press Enter:
   ```
   python --version or py --version
   ```
3. If you see something like `Python 3.10.12`, Python is already installed — skip to Section 3.3.
4. If you see an error or `Python was not found`, continue to Section 3.2.

**On Mac:**
1. Open **Terminal** (search for it in Spotlight with `Cmd + Space`).
2. Type `python3 --version` and press Enter.
3. If you see a version number of 3.9 or higher, skip to Section 3.3.

### 3.2 Install Python

1. Go to the official Python download page:
   **[https://www.python.org/downloads/](https://www.python.org/downloads/)**
2. Click the large yellow **`Download Python 3.x.x`** button (the latest version shown is fine).
3. Run the downloaded installer.
4. **Important (Windows only):** On the first screen of the installer, make sure to check the box that says **"Add Python to PATH"** before clicking Install Now.
5. Follow the remaining prompts and complete the installation.
6. To verify the installation, open a new Command Prompt and type `python --version` or `py --version` again.

### 3.3 Install the required libraries

NüshuSight depends on several Python libraries. You only need to install these once.

**On Windows:**
1. Open Command Prompt (`Win + R` → type `cmd` → Enter).
2. Copy and paste the following command, then press Enter:
   ```
   pip install flask pillow opencv-python torch torchvision
   ```
3. Wait for the installation to complete. This may take a few minutes depending on your internet speed.

**On Mac:**
1. Open Terminal.
2. Run:
   ```
   pip3 install flask pillow opencv-python torch torchvision
   ```

> **Note:** If you see a warning saying `pip is not recognized`, try replacing `pip` with `py -m pip` in the command above.

> **Note:** If you have a dedicated NVIDIA GPU and wish to use it for faster processing, visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for GPU-specific installation instructions. For most users, the standard CPU installation above is sufficient.

### 3.4 Compatible browsers

NüshuSight's interface runs in your web browser. The following browsers are supported:

| Browser | Support |
|---------|---------|
| Google Chrome | ✓ Recommended |
| Microsoft Edge | ✓ Recommended |
| Mozilla Firefox | ✓ Supported |
| Safari | ✗ Not recommended |

---

## 4. Start the Application

### Step 1 — Open a terminal in the project folder

**On Windows:**
1. Open File Explorer and navigate to the `NushuSight` folder.
2. Click on the address bar at the top of the window (where the folder path is shown).
3. Type `cmd` and press Enter. A Command Prompt window will open, already pointing to the correct folder.

Alternatively, press `Win + R`, type `cmd`, press Enter, then navigate to your folder:
```
cd C:\path\to\NushuSight
```
(Replace `C:\path\to\NushuSight` with the actual path to your folder.)

**On Mac:**
1. Open Terminal.
2. Type `cd ` (with a space), then drag the `NushuSight` folder from Finder into the Terminal window, and press Enter.

### Step 2 — Run the application

In the terminal, type the following command and press Enter:

```bash
python server/app_v2.py --port 5000 or py server/app_v2.py --port 5000
```

On Mac, use `python3` instead of `python`:
```bash
python3 server/app_v2.py --port 5000
```

The first launch will take a few seconds as the model loads. When you see the following message, the application is ready:

```
 * Serving Flask app 'app_v2'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Step 3 — Open the interface

Open your browser and go to:

```
http://127.0.0.1:5000
```

The NüshuSight OCR Studio interface will appear in your browser.

> **Note:** The terminal window must stay open while you use the application. Do not close it.

### Stopping the application

When you are done, go back to the terminal window and press `Ctrl + C` to stop the server.

---

## 5. Usage Overview

NüshuSight guides you through a 4-step workflow:

| Step | Description |
|------|-------------|
| **Step 1** | Upload a photo or scanned image of a Nüshu manuscript |
| **Step 2** | Review the automatically detected character bounding boxes; adjust or add boxes if needed |
| **Step 3** | For each detected character, review the Top-5 recognition candidates and confirm or correct the result |
| **Step 4** | Export the final results as a Unicode text string and a translation table (character image, Nüshu, corresponding Hanzi, pronunciation) |

The interface supports both **Chinese (中文)** and **English** — use the language toggle button in the top bar to switch.

---

## Troubleshooting

**"ModuleNotFoundError" when starting**
Run the pip install command again (Section 3.3) to make sure all libraries are installed.

**The browser shows "This site can't be reached"**
Make sure the terminal is still running (you should see the "OCR Studio v2 running" message). Try refreshing the page.

**Recognition accuracy is low**
- Make sure the image is well-lit and the characters are clearly visible.
- In Step 2, check that each bounding box tightly encloses a single character.
- Use Step 3 to manually select the correct character from the Top-5 candidates when needed.

---

---

# 中文使用说明

## 一、下载软件代码

### 方式 A — 下载 ZIP 压缩包（推荐，无需安装 Git）

1. 用浏览器打开：
   **[https://github.com/EllieWei2008/NushuSight](https://github.com/EllieWei2008/NushuSight)**
2. 点击页面右上方的绿色 **`Code`** 按钮。
3. 在弹出菜单中点击 **`Download ZIP`**。
4. 将下载的 ZIP 文件保存到电脑，例如桌面。
5. 右键点击 ZIP 文件，选择 **"解压全部"**（Windows）或直接双击解压（Mac）。
6. 解压后会得到一个名为 `NushuSight-main` 的文件夹，可将其重命名为 `NushuSight`。

### 方式 B — 使用 Git 克隆（适合熟悉 Git 的用户）

```bash
git clone https://github.com/EllieWei2008/NushuSight.git
```

---

## 二、下载识别模型权重文件

识别模型文件（`oneshot_best.pth`，约 106 MB）由于体积较大，无法直接存放在代码仓库中，需要单独下载后手动放置到指定位置。

**第一步：下载文件**

1. 打开发布页面：
   **[https://github.com/EllieWei2008/NushuSight/releases](https://github.com/EllieWei2008/NushuSight/releases)**
2. 点击列表最上方的最新版本。
3. 在 **Assets**（附件）区域，点击 **`oneshot_best.pth`** 进行下载。

**第二步：将文件放到正确的位置**

在 `NushuSight` 文件夹中，找到 `checkpoints` 子文件夹，其中还有一个 `oneshot` 子文件夹，将下载的文件放入其中：

```
NushuSight/
└── checkpoints/
    └── oneshot/
        └── oneshot_best.pth   ← 将下载的文件放在这里
```

> 如果 `checkpoints/oneshot/` 文件夹不存在，请手动创建。

---

## 三、准备运行环境

NüshuSight 基于 **Python** 运行。Python 是一个免费的编程语言运行环境，您无需编写任何代码，只需安装它来运行本工具。此外还需要一个兼容的浏览器。

### 3.1 检查 Python 是否已安装

**Windows 系统：**
1. 按下 `Win + R`，输入 `cmd`，按回车，打开命令提示符窗口。
2. 输入以下命令并按回车：
   ```
   python --version 或者 py --version
   ```
3. 如果显示类似 `Python 3.10.12` 的版本号，说明 Python 已安装，跳到第 3.3 节。
4. 如果提示找不到命令或报错，请继续第 3.2 节进行安装。

**Mac 系统：**
1. 打开 **终端**（按 `Cmd + Space` 搜索"终端"）。
2. 输入 `python3 --version` 并按回车。
3. 如果显示 3.9 或更高版本，跳到第 3.3 节。

### 3.2 安装 Python

1. 打开 Python 官方下载页面：
   **[https://www.python.org/downloads/](https://www.python.org/downloads/)**
2. 点击页面上的黄色大按钮 **`Download Python 3.x.x`**（显示的最新版本即可）。
3. 运行下载的安装程序。
4. **Windows 用户特别注意：** 安装程序的第一个界面，务必勾选 **"Add Python to PATH"** 这个选项，再点击 Install Now。否则后续步骤可能无法识别 Python 命令。
5. 按照提示完成安装。
6. 安装完成后，重新打开一个命令提示符窗口，输入 `python --version` 或者 `py --version` 验证安装是否成功。

### 3.3 安装所需依赖库

NüshuSight 依赖几个 Python 库，只需安装一次即可。

**Windows 系统：**
1. 打开命令提示符（`Win + R` → 输入 `cmd` → 回车）。
2. 复制以下命令，粘贴到命令提示符中，按回车执行：
   ```
   pip install flask pillow opencv-python torch torchvision
   ```
3. 等待安装完成，根据网络速度可能需要几分钟。

**Mac 系统：**
1. 打开终端。
2. 运行：
   ```
   pip3 install flask pillow opencv-python torch torchvision
   ```

> **提示：** 如果提示 `pip 不是内部或外部命令`，请将命令中的 `pip` 替换为 `py -m pip` 再试。

> **提示：** 如果您的电脑配备了 NVIDIA 显卡并希望使用 GPU 加速推理，请访问 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 获取对应的安装命令。对大多数用户而言，上述标准安装（CPU 版本）已足够使用。

### 3.4 兼容浏览器

NüshuSight 的操作界面在浏览器中运行，支持以下浏览器：

| 浏览器 | 支持情况 |
|--------|---------|
| Google Chrome（谷歌浏览器）| ✓ 推荐 |
| Microsoft Edge（微软浏览器）| ✓ 推荐 |
| Mozilla Firefox（火狐浏览器）| ✓ 支持 |
| Safari | ✗ 不推荐 |

---

## 四、启动软件

### 第一步：在软件目录下打开命令行窗口

**Windows 系统：**
1. 打开文件资源管理器，找到 `NushuSight` 文件夹。
2. 点击窗口顶部的地址栏（显示文件夹路径的地方）。
3. 直接输入 `cmd` 并按回车，即可打开一个已定位到该文件夹的命令提示符窗口。

或者，按 `Win + R`，输入 `cmd` 打开命令提示符，然后输入：
```
cd C:\你的路径\NushuSight
```
（将 `C:\你的路径\NushuSight` 替换为实际的文件夹路径）

**Mac 系统：**
1. 打开终端。
2. 输入 `cd `（注意末尾有一个空格），然后将 `NushuSight` 文件夹从 Finder 拖入终端窗口，按回车。

### 第二步：运行软件

在命令行窗口中输入以下命令并按回车：

```bash
python server/app_v2.py --port 5000 或者 py server/app_v2.py --port 5000
```

Mac 用户请使用 `python3`：
```bash
python3 server/app_v2.py --port 5000
```

首次启动需要几秒钟加载模型。当看到以下提示时，说明软件已准备好：

```
 * Serving Flask app 'app_v2'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### 第三步：打开操作界面

打开浏览器，在地址栏输入：

```
http://127.0.0.1:5000
```

NüshuSight OCR Studio 界面将在浏览器中显示。

> **注意：** 使用软件期间，命令行窗口必须保持开启，请勿关闭。

### 停止软件

使用完毕后，切换到命令行窗口，按 `Ctrl + C` 停止服务。

---

## 五、使用流程说明

NüshuSight 采用四步流程：

| 步骤 | 说明 |
|------|------|
| **第一步** | 上传女书手稿的照片或扫描图像 |
| **第二步** | 查看自动检测的字符框，根据需要调整或补充 |
| **第三步** | 对每个字符查看 Top-5 识别候选项，确认或人工选择正确结果 |
| **第四步** | 导出识别结果，包括 Unicode 文本及对照表（字符图、女书、对应汉字、发音） |

界面支持**中文**和 **English** 双语切换，点击顶栏的语言切换按钮即可。

---

## 常见问题

**启动时报 "ModuleNotFoundError"**
请重新执行第三节的 pip 安装命令，确保所有依赖库已正确安装。

**浏览器显示"无法访问此网站"**
请确认命令行窗口仍在运行（应看到"OCR Studio v2 running"的提示）。刷新页面后重试。

**识别率偏低**
- 确保图像清晰、光线充足，字符轮廓完整可见。
- 在第二步中检查每个字符框是否紧贴单个字符。
- 在第三步中，当 Top-1 结果不正确时，可从 Top-5 候选中手动选择正确字符。

---

## 许可证

MIT License. 详见 [LICENSE](LICENSE)。
