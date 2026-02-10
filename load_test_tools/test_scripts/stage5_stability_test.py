#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段5: 稳定性测试
目标: 验证基本稳定性，70-80%负载持续10分钟
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
from core.load_test_runner import run_single_test

# ==================== 测试配置（硬编码） ====================
BASE_CONFIG = {
    'url': 'https://xxxx.com/ner_extract_info/api/extract',
    'timeout': 5,
    't1': 1.0,
    't2': 3.0,
    'content': "广东省深圳市龙岗区坂田街道长坑路",
}

# 阶段5特定配置: 70-80%负载持续10分钟
# 假设拐点为50
INFLECTION_POINT = 50
STABLE_LOAD_RATIO = 0.75  # 70-80%负载
DURATION = 600  # 10分钟

TEST_CONFIG = {
    'concurrent': int(INFLECTION_POINT * STABLE_LOAD_RATIO),
    'duration': DURATION
}
# 结果: 并发37持续10分钟

# 输出目录
OUTPUT_DIR = 'test/load_test/reports/stage5'
SAVE_JSON = False

# =========================================================

async def main():
    """主函数"""
    print(f"\n{'='*80}")
    print("阶段5: 稳定性测试")
    print(f"{'='*80}")
    print(f"测试目标: 验证基本稳定性，70-80%负载持续10分钟")
    print(f"测试配置: 并发={TEST_CONFIG['concurrent']}, 持续时间={TEST_CONFIG['duration']}秒")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    try:
        result = await run_single_test(
            url=BASE_CONFIG['url'],
            concurrent=TEST_CONFIG['concurrent'],
            duration=TEST_CONFIG['duration'],
            payload={"Content": BASE_CONFIG['content']},
            timeout=BASE_CONFIG['timeout'],
            t1=BASE_CONFIG['t1'],
            t2=BASE_CONFIG['t2'],
            output_dir=OUTPUT_DIR,
            save_json=SAVE_JSON,
            test_name="stability_test"
        )

        # 打印结果摘要
        stats = result.get_statistics()
        print(f"\n{'='*80}")
        print("阶段5测试完成！")
        print(f"{'='*80}")
        print(f"QPS: {stats.get('qps', 0):.2f}")
        print(f"Fast率: {stats.get('fast_rate', 0):.2f}%")
        print(f"Slow率: {stats.get('slow_rate', 0):.2f}%")
        print(f"Bad率: {stats.get('bad_rate', 0):.2f}%")

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
