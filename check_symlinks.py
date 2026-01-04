#!/usr/bin/env python3
"""
检查并修复项目中的符号链接问题（Windows 兼容性）

此脚本会：
1. 检查项目中是否存在符号链接
2. 如果发现符号链接，提供修复建议
"""

import os
import sys
from pathlib import Path

def find_symlinks(root_dir):
    """查找所有符号链接"""
    symlinks = []
    root_path = Path(root_dir)
    
    for path in root_path.rglob('*'):
        if path.is_symlink():
            symlinks.append(path)
    
    return symlinks

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"检查目录: {root_dir}")
    print("-" * 60)
    
    symlinks = find_symlinks(root_dir)
    
    if symlinks:
        print(f"发现 {len(symlinks)} 个符号链接:")
        for symlink in symlinks:
            target = os.readlink(symlink)
            print(f"  - {symlink}")
            print(f"    指向: {target}")
        print("\n警告: 这些符号链接在 Windows 上可能导致问题！")
        print("建议: 将这些符号链接替换为实际文件副本。")
        return 1
    else:
        print("✓ 未发现符号链接，项目应该可以在 Windows 上正常使用。")
        return 0

if __name__ == "__main__":
    sys.exit(main())
