#!/usr/bin/env python
# =============================================================================
# 子图去重脚本
# =============================================================================
# 用途：遍历工作目录下所有子图的 model.py，计算 SHA256 哈希生成 graph_hash.txt，
#       找出内容完全相同的子图并生成去重报告，支持一键删除重复目录。
#
# 为什么需要去重：
#   同一基础模型的微调变体（fine-tune variants）通常共享完全相同的计算图，
#   只是权重不同。去重后体积可缩减 90%+（实测：85K 子图 -> 1.5K 唯一子图，
#   2.3 GB -> 172 MB）。
#
# 用法：
#   python gen_hash_and_dedup.py <workspace_dir> [--remove]
#
# 参数：
#   workspace_dir   抽取结果目录（含 model.py 子图），默认当前目录 "."
#   --remove        可选，确认后直接删除重复子图目录
#
# 示例：
#   # 1. 仅分析，不删除
#   python gen_hash_and_dedup.py ./success_20260515_merged
#
#   # 2. 分析并直接删除重复目录（保留每组第一个）
#   python gen_hash_and_dedup.py ./success_20260515_merged --remove
#
# 输出示例：
#   Found 6919 model.py files under ./success_20260515_merged
#   Progress: 2000/6919
#   ...
#
#   Step 1 - Generate graph_hash.txt:
#     Total model.py: 6919
#     Generated/Updated: 6919
#     Failed: 0
#
#   Step 2 - Deduplication analysis:
#     Total subgraphs: 6919
#     Unique graphs: 1486
#     Duplicate groups: 421
#     Subgraphs involved in duplication: 5433
#     Can be removed (keeping one per group): 5012
#
#     Dedup report saved to: ./success_20260515_merged/dedup_report.txt
#
#   To remove duplicates, run:
#     python /path/to/gen_hash_and_dedup.py ./success_20260515_merged --remove
#
#   # 若加了 --remove，还会输出：
#   Removed 5012 duplicate subgraph directories.
#   Remaining subgraphs: 1907
#
# 输出文件：
#   <workspace>/dedup_report.txt    去重报告，包含每组重复的 hash、数量和路径
#   <subgraph_dir>/graph_hash.txt   每个子图目录下的哈希文件
#
# 依赖：Python 3.6+
# =============================================================================

import os
import sys
from collections import defaultdict

# Ensure graph_net is importable when running this script standalone
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRAPHNET_ROOT = os.path.join(_SCRIPT_DIR, "..", "..", "..")
if _GRAPHNET_ROOT not in sys.path:
    sys.path.insert(0, _GRAPHNET_ROOT)

from graph_net.hash_util import get_sha256_hash  # noqa: E402


def find_model_files(workspace):
    results = []
    for root, dirs, files in os.walk(workspace):
        if "model.py" in files:
            results.append(os.path.join(root, "model.py"))
    return sorted(results)


def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."

    # Step 1: Find all model.py and generate graph_hash.txt
    generated = 0
    failed = 0
    hash_to_paths = defaultdict(list)

    model_files = find_model_files(workspace)
    total = len(model_files)
    print("Found %d model.py files under %s" % (total, workspace))

    for i, model_py in enumerate(model_files, 1):
        subgraph_dir = os.path.dirname(model_py)
        hash_file = os.path.join(subgraph_dir, "graph_hash.txt")

        try:
            with open(model_py, "r") as f:
                model_code = f.read()
            file_hash = get_sha256_hash(model_code)

            with open(hash_file, "w") as f:
                f.write(file_hash)
            hash_to_paths[file_hash].append(subgraph_dir)
            generated += 1

        except Exception as e:
            failed += 1
            print("  [FAIL] %s: %s" % (subgraph_dir, e))

        if i % 2000 == 0:
            print("  Progress: %d/%d" % (i, total))

    print("\nStep 1 - Generate graph_hash.txt:")
    print("  Total model.py: %d" % total)
    print("  Generated/Updated: %d" % generated)
    print("  Failed: %d" % failed)

    # Step 2: Dedup analysis
    unique_count = len(hash_to_paths)
    dup_groups = dict(
        (h, paths) for h, paths in hash_to_paths.items() if len(paths) > 1
    )
    dup_subgraph_count = sum(len(p) for p in dup_groups.values())
    removable = dup_subgraph_count - len(dup_groups)

    print("\nStep 2 - Deduplication analysis:")
    print("  Total subgraphs: %d" % total)
    print("  Unique graphs: %d" % unique_count)
    print("  Duplicate groups: %d" % len(dup_groups))
    print("  Subgraphs involved in duplication: %d" % dup_subgraph_count)
    print("  Can be removed (keeping one per group): %d" % removable)

    # Step 3: Write dedup report
    report_path = os.path.join(workspace, "dedup_report.txt")
    with open(report_path, "w") as f:
        f.write("Deduplication Report\n")
        f.write("====================\n")
        f.write("Total subgraphs: %d\n" % total)
        f.write("Unique graphs: %d\n" % unique_count)
        f.write("Duplicate groups: %d\n" % len(dup_groups))
        f.write("Removable duplicates: %d\n\n" % removable)

        for h, paths in sorted(dup_groups.items(), key=lambda x: -len(x[1])):
            f.write("Hash: %s\n" % h)
            f.write("  Count: %d\n" % len(paths))
            f.write("  Keep:   %s\n" % paths[0])
            for p in paths[1:]:
                f.write("  Remove: %s\n" % p)
            f.write("\n")

    print("\n  Dedup report saved to: %s" % report_path)

    # Step 4: Ask before removing
    if len(sys.argv) > 2 and sys.argv[2] == "--remove":
        import shutil

        removed = 0
        for h, paths in dup_groups.items():
            for p in paths[1:]:
                if os.path.exists(p):
                    shutil.rmtree(p)
                    removed += 1
        print("\nRemoved %d duplicate subgraph directories." % removed)
        print("Remaining subgraphs: %d" % (total - removed))
    else:
        print("\nTo remove duplicates, run:")
        print("  python %s %s --remove" % (os.path.abspath(__file__), workspace))


if __name__ == "__main__":
    main()
