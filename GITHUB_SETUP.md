# GitHub 上传指南 - 解决 Windows 符号链接问题

## 为什么上传到 GitHub 可以解决问题？

GitHub 在克隆仓库时，默认情况下会：
1. 将符号链接转换为普通文件（如果配置正确）
2. 使用 `.gitattributes` 文件控制文件处理方式
3. Windows 用户可以通过 `git config core.symlinks false` 禁用符号链接创建

## 快速开始（推荐）

使用提供的脚本快速初始化：

```bash
cd /path/to/TienKung-Lab_other0103
./setup_git.sh
```

脚本会自动：
- 检查符号链接
- 初始化 Git 仓库
- 配置 Git 用户信息
- 添加所有文件
- 创建初始提交

## 手动初始化 Git 仓库并上传到 GitHub

### 步骤 1: 初始化 Git 仓库

```bash
cd /path/to/TienKung-Lab_other0103

# 初始化 Git 仓库
git init

# 配置 Git（可选，但推荐）
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 步骤 2: 创建 .gitignore 文件（如果还没有）

确保 `.gitignore` 文件包含以下内容：

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# 其他
*.pt
*.onnx
```

### 步骤 3: 验证没有符号链接

```bash
# 运行检查脚本
python check_symlinks.py
```

应该显示：`✓ 未发现符号链接，项目应该可以在 Windows 上正常使用。`

### 步骤 4: 添加文件并提交

```bash
# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: Fixed symbolic links for Windows compatibility"
```

### 步骤 5: 在 GitHub 上创建仓库

1. 登录 GitHub
2. 点击右上角的 "+" → "New repository"
3. 输入仓库名称（如 `TienKung-Lab`）
4. 选择 Public 或 Private
5. **不要**勾选 "Initialize this repository with a README"（因为本地已有文件）
6. 点击 "Create repository"

### 步骤 6: 连接本地仓库到 GitHub

```bash
# 添加远程仓库（替换 YOUR_USERNAME 和 YOUR_REPO_NAME）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 或者使用 SSH（如果已配置）
# git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# 推送代码
git branch -M main
git push -u origin main
```

## Windows 用户克隆时的配置

Windows 用户在克隆仓库时，可以运行以下命令来避免符号链接问题：

```bash
# 配置 Git 不创建符号链接
git config core.symlinks false

# 然后克隆仓库
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

或者，Windows 用户可以在全局配置中设置：

```bash
git config --global core.symlinks false
```

## 验证上传是否成功

1. 在 GitHub 网页上检查文件是否存在
2. 在 Windows 机器上测试克隆：
   ```bash
   git config core.symlinks false
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```
3. 检查克隆后的文件是否为普通文件（不是符号链接）

## 注意事项

- ✅ 所有符号链接已替换为实际文件
- ✅ `.gitattributes` 文件已配置，确保文件被正确处理
- ✅ 上传到 GitHub 后，Windows 用户可以通过 `git config core.symlinks false` 安全克隆

## 如果仍然遇到问题

如果 Windows 用户在克隆后仍然遇到符号链接问题：

1. 确保他们运行了 `git config core.symlinks false`
2. 检查 `.gitattributes` 文件是否在仓库中
3. 尝试删除本地克隆，重新克隆：
   ```bash
   rm -rf TienKung-Lab_other0103
   git config core.symlinks false
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```

