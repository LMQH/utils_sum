#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
负责配置文件的加载、验证和合并
"""

import os
import json
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典，如果文件不存在或读取失败则返回空字典
    """
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"警告: 无法读取配置文件 {config_path}: {str(e)}")
        return {}


def validate_config(config: Dict) -> bool:
    """
    验证配置完整性

    Args:
        config: 配置字典

    Returns:
        验证是否通过

    Raises:
        ValueError: 当配置验证失败时
    """
    # 必需参数检查
    if 'url' not in config:
        raise ValueError("配置文件缺少必需参数: url")

    # 互斥参数检查
    if config.get('total') and config.get('duration'):
        raise ValueError("total 和 duration 不能同时设置")

    return True


def merge_config(
    file_config: Dict,
    cli_config: Dict,
    defaults: Dict
) -> Dict:
    """
    合并配置：默认值 -> 配置文件 -> 命令行参数

    Args:
        file_config: 从配置文件读取的配置
        cli_config: 命令行参数配置
        defaults: 默认配置

    Returns:
        合并后的配置
    """
    merged = defaults.copy()
    merged.update(file_config)
    merged.update(cli_config)
    return merged


def get_default_config() -> Dict:
    """
    获取默认配置

    Returns:
        默认配置字典
    """
    return {
        'concurrent': 10,
        'timeout': 5,
        't1': 1.0,
        't2': 3.0,
        'content': "广东省深圳市龙岗区坂田街道长坑路西2巷2号202 黄大大 18273778575",
        'output_dir': 'test/load_test/reports',
        'json': False,
        'batch_mode': {
            'cooldown': 5
        }
    }
