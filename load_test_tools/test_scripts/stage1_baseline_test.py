#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1: 基准测试
目标: 获取单请求最优性能，测试低并发下的表现
"""

import asyncio
import sys
import os

# 动态添加 test/load_test 目录到 Python 路径
# 这样无论项目根目录结构如何，只要 test/load_test 目录结构不变就能工作
script_dir = os.path.dirname(os.path.abspath(__file__))
load_test_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, load_test_dir)

# 直接导入，不依赖项目根目录的包结构
from core.load_test_runner import run_sequential_tests

# ==================== 测试配置（硬编码） ====================
BASE_CONFIG = {
    'url': 'https://xxxx.com/ner_extract_info/api/extract',
    'timeout': 5,
    't1': 1.0,
    't2': 3.0,
    'content': "广东省深圳市龙岗区坂田街道长坑路",
}

# 阶段1特定配置: 并发1-5，各100请求
TEST_CONFIGS = [
    {'concurrent': 1, 'total': 100},
    {'concurrent': 3, 'total': 100},
    {'concurrent': 5, 'total': 100},
]

# 输出目录
OUTPUT_DIR = 'test/load_test/reports/stage1'
SAVE_JSON = False
COOLDOWN = 5  # 测试间隔（秒）

# =========================================================

async def main():
    """主函数"""
    print(f"\n{'='*80}")
    print("阶段1: 基准测试")
    print(f"{'='*80}")
    print(f"测试目标: 获取单请求最优性能，测试低并发下的表现")
    print(f"测试配置数: {len(TEST_CONFIGS)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    try:
        results = await run_sequential_tests(
            test_configs=TEST_CONFIGS,
            base_config=BASE_CONFIG,
            output_dir=OUTPUT_DIR,
            save_json=SAVE_JSON,
            cooldown=COOLDOWN,
            generate_summary=True
        )

        print(f"\n{'='*80}")
        print("阶段1测试完成！")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
