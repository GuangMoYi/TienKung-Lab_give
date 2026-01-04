#!/bin/bash
# Git 仓库初始化脚本
# 用于准备将项目上传到 GitHub

set -e

echo "=========================================="
echo "Git 仓库初始化脚本"
echo "=========================================="
echo ""

# 检查是否已经是 Git 仓库
if [ -d ".git" ]; then
    echo "⚠️  警告: 这已经是一个 Git 仓库"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查符号链接
echo "1. 检查符号链接..."
if command -v python3 &> /dev/null; then
    python3 check_symlinks.py
    if [ $? -ne 0 ]; then
        echo "❌ 发现符号链接，请先修复后再继续"
        exit 1
    fi
else
    echo "⚠️  无法运行 Python 检查脚本，跳过..."
fi

echo ""
echo "2. 初始化 Git 仓库..."
git init

echo ""
echo "3. 配置 Git（可选）..."
read -p "输入您的 Git 用户名（直接回车跳过）: " git_name
if [ ! -z "$git_name" ]; then
    git config user.name "$git_name"
fi

read -p "输入您的 Git 邮箱（直接回车跳过）: " git_email
if [ ! -z "$git_email" ]; then
    git config user.email "$git_email"
fi

echo ""
echo "4. 添加文件..."
git add .

echo ""
echo "5. 创建初始提交..."
git commit -m "Initial commit: Fixed symbolic links for Windows compatibility"

echo ""
echo "=========================================="
echo "✅ Git 仓库初始化完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 在 GitHub 上创建新仓库"
echo "2. 运行以下命令连接并推送："
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "详细说明请查看 GITHUB_SETUP.md"

