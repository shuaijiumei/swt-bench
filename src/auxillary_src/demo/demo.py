import os
import sys

# 添加上级目录到 sys.path 以便导入 extract_patches
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from extract_patches import extract_minimal_patch

def main():
    # 示例 patch 内容，可以根据需要替换成实际 diff 文本
    sample_patch = """diff --git a/sample.txt b/sample.txt
--- a/sample.txt
+++ b/sample.txt
@@ -1,3 +1,3 @@
-foo
+bar
 baz
"""
    new_patch = extract_minimal_patch(sample_patch)
    print("Extracted Minimal Patch:\n", new_patch)

if __name__ == "__main__":
    main()
