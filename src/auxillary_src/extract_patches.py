import dataclasses
import os
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Dict
from collections import defaultdict
import ast
import textwrap
import pathlib
import editdistance
import argparse
import git
import re

### MARK - Patch Correction
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)


@dataclasses.dataclass
class FuzzyPatch:
    rough_line_number: int
    preceding_lines: List[str]
    deleted_lines: List[str]
    inserted_lines: List[str]
    following_lines: List[str]


@dataclasses.dataclass
class FuzzyFilePatch:
    file_name: str
    diffs: List[FuzzyPatch]


@dataclasses.dataclass
class CustomPatch:
    file_name: str
    patch_type: str
    rough_line_number: str
    changed_lines: list


# 函数：get_first_idx
# 功能：返回列表中第一个出现 "-" 或 "+" 的索引；如果都不存在，则返回列表长度
# 参数：charlist - 一个字符列表
def get_first_idx(charlist):
    """Get index of first occurrence of "-" or "+" in charlist"""
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)


# 函数：get_last_idx
# 功能：返回列表中最后一个 "-" 或 "+" 出现的位置的后续索引
# 参数：charlist - 一个字符列表
def get_last_idx(charlist):
    """Get index of last occurrence of "-" or "+" in charlist"""
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1


# 函数：strip_content
# 功能：清理 hunk 内容，移除非 "-" 或 "+" 开头的行以及各行尾部空白字符
# 参数：hunk - 表示 diff hunk 的字符串
# 返回：清理后的 hunk 字符串及调整量（用于修正起始行号）
def strip_content(hunk):
    """Remove trailing non +/- lines and trailing whitespace per line per hunk"""
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1

# 函数：remove_binary_diffs
# 功能：过滤掉 diff 文本中针对二进制文件产生的差异内容
# 参数：diff_content - 原始 diff 文本字符串
# 返回：不包含二进制文件数据的 diff 文本
def remove_binary_diffs(diff_content):
    binary_file_indicator = 'Binary files'

    lines = diff_content.splitlines()

    new_lines = []
    curr_diff = []
    skip_current_diff = False
    for line in lines + ["diff --git"]:
        if line.startswith('diff --git'):
            if curr_diff and not skip_current_diff:
                new_lines.append('\n'.join(curr_diff))
            curr_diff = []
            skip_current_diff = False
        if binary_file_indicator in line:
            skip_current_diff = True
        curr_diff.append(line)

    return '\n'.join(new_lines) + "\n"

# 函数：extract_fuzzy_patch
# 功能：提取模糊 patch，清理 hunk 空白及前置、后置上下文，生成 FuzzyFilePatch 列表
# 参数：model_patch - 原始 diff 文本
# 返回：FuzzyFilePatch 对象列表，每个对象包含文件名及其模糊 patch 信息
def extract_fuzzy_patch(model_patch) -> List[FuzzyFilePatch]:
    """
    Wrapper function that takes hunk and
    * Removes trailing whitespace per line per hunk
    * Returns new patch
    """
    model_patch = model_patch.lstrip("\n")
    patches = []
    for patch in PATCH_PATTERN.findall(model_patch):
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        subpatches = []
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content_lines = content.split("\n")
            content_lines = content_lines[1:]
            i = 0
            deleted_lines = []
            inserted_lines = []
            preceding_lines = []
            following_lines = []
            change_start = pre_start
            # positions: 0 = preceding, 1 = in change, 2 = following
            position = 0
            for i, line in enumerate(content_lines):
                if line.startswith(" "):
                    if position == 1:
                        position = 2
                    if position == 0:
                        preceding_lines.append(line[1:])
                    if position == 2:
                        following_lines.append(line[1:])
                if line.startswith("-"):
                    if position == 2:
                        subpatches.append(FuzzyPatch(change_start, preceding_lines, deleted_lines, inserted_lines, following_lines))
                        change_start = pre_start + i - len(following_lines)
                        preceding_lines = following_lines
                        following_lines = []
                        deleted_lines = []
                        inserted_lines = []
                        position = 1
                    if position == 0:
                        position = 1
                    deleted_lines.append(line[1:])
                if line.startswith("+"):
                    if position == 2:
                        subpatches.append(FuzzyPatch(change_start, preceding_lines, deleted_lines, inserted_lines, following_lines))
                        change_start = pre_start + i - len(following_lines)
                        preceding_lines = following_lines
                        following_lines = []
                        deleted_lines = []
                        inserted_lines = []
                        position = 1
                    if position == 0:
                        position = 1
                    inserted_lines.append(line[1:])
                if line.startswith("```") or line.startswith("<"):
                    break
            subpatches.append(FuzzyPatch(change_start, preceding_lines, deleted_lines, inserted_lines, following_lines))
            content = "\n".join(content_lines[:i])
        patches.append(FuzzyFilePatch(patch_header.split()[1][2:], subpatches))

    return patches

# 函数：extract_custom_patches
# 功能：从原始 diff 文本中提取自定义 patch，识别文件名、类型、行号和变更内容
# 参数：model_patch - 包含 diff 文本的字符串
# 返回：CustomPatch 对象列表，每个对象描述一次自定义 patch 变更
def extract_custom_patches(model_patch):
    """
    Wrapper function that takes response and
    - searches for all lines with "diff"
    - extracts the file name, file line start and end and changed lines
    """
    model_patch = model_patch.lstrip("\n").splitlines()
    patches = []
    for i, line in enumerate(model_patch):
        if line.startswith("diff"):
            try:
                file_name = model_patch[i+1]
                patch_type = model_patch[i+2]
                rough_line_number = model_patch[i+3]
                j = i + 4
                for j in range(i+4, len(model_patch)):
                    if model_patch[j].startswith("end diff"):
                        break
                changed_lines = model_patch[i + 4:j]
            except:
                continue
            patches.append(CustomPatch(file_name, patch_type, rough_line_number, changed_lines))
    return patches

# 函数：extract_minimal_patch
# 功能：生成最精简的 diff patch，处理 hunk 内容并重新计算行号和差异行数
# 参数：model_patch - 原始 diff 文本字符串
# 返回：处理后的最小化 diff patch 字符串
def extract_minimal_patch(model_patch) -> str:
    """
    Wrapper function that takes hunk and
    * Removes trailing non +/- lines and trailing whitespace per line per hunk
    * Recalculates hunk start/end position and diff delta
    * Returns new patch
    """
    model_patch = remove_binary_diffs(model_patch)
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


# 函数：get_hunk_stats
# 功能：根据 hunk 内的内容统计上下文、增加和删除的行数，更新 hunk 的起始行、长度以及累计偏移量
# 参数：pre_start, pre_len, post_start, post_len - 原始 hunk 的起始行和长度；hunk - 当前 hunk 的文本；total_delta - 累计偏移量
# 返回：更新后的 pre_start, pre_len, post_start, post_len 和 total_delta
def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    """Recalculate hunk start/end position and diff delta"""
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta

# 函数：apply_fuzzy_patches
# 功能：将提取到的模糊 patch 应用到指定的测试路径下，匹配时不要求精确行号
# 参数：
#   fuzzy_patch - FuzzyFilePatch 对象列表
#   testbed - 测试环境目录路径
#   patch_type - patch 类型（默认为 "fuzzy"）
# 返回：布尔值，表示 patch 是否成功应用
def apply_fuzzy_patches(fuzzy_patch: List[FuzzyFilePatch], testbed: str, patch_type: str = "fuzzy") -> bool:
    """
    Apply a git diff patch without exact line number matching

    Args:
        fuzzy_patches (list): list of patches to apply
        patch_type (str): Type of patch (e.g. "eval", "test")
    Returns:
        bool: whether the patch applied successfully
    """
    if not fuzzy_patch:
        return False

    # Apply patch to testbed directory
    for patch in fuzzy_patch:
        file_name = patch.file_name
        os.path.join(testbed, file_name)
        try:
            with open(file_name, "r") as f:
                file = f.read()
        except FileNotFoundError as e:
            print(f"Patch file not found ({file_name} for patch type {patch_type})")
            return False

        lines = file.splitlines()
        for diff in patch.diffs:
            # find position in the file where the patch should be applied
            best_start = 0
            best_start_score = 0
            for i, line in enumerate(lines):
                score = overlap_score(diff.preceding_lines + diff.deleted_lines, lines[i:])
                if score > best_start_score:
                    best_start_score = score
                    best_start = min(i + len(diff.preceding_lines), len(lines))

            # find position of the last line of the patch
            best_end = len(lines)
            best_end_score = 0
            for i, line in enumerate(lines):
                score = overlap_score(diff.following_lines, lines[i:])
                if score > best_end_score:
                    best_end_score = score
                    best_end = i

            if best_end < best_start:
                print(f"Invalid patch reverses ({file_name} for patch type {patch_type})")

            # apply the patch
            lines = lines[:best_start] + diff.inserted_lines + lines[best_end:]

        with open(file_name, "w") as f:
            f.write("\n".join(lines))

    # Patch apply succeeded
    print(f"Custom patch successful ({file_name} for patch type {patch_type})")
    return True


class ReplaceFunctionTransformer(ast.NodeTransformer):
    def __init__(self, new_ast, approximate_lineno):
        self.new_ast = new_ast
        self.approximate_lineno = approximate_lineno
        self.any_change_applied = False

    def visit_FunctionDef(self, node):
        if isinstance(node, ast.FunctionDef) and isinstance(self.new_ast, ast.FunctionDef) and node.name == self.new_ast.name:
            self.any_change_applied = True
            return self.new_ast
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if isinstance(node, ast.ClassDef) and isinstance(self.new_ast, ast.ClassDef) and node.name == self.new_ast.name:
            self.any_change_applied = True
            return self.new_ast
        return self.generic_visit(node)


# 函数：apply_custom_patches
# 功能：将自定义 patch 应用到任务环境中，并返回 git patch
# 参数：
#   custom_patches - CustomPatch 对象列表
#   testbed - 测试环境目录路径
#   patch_type - patch 类型（默认为 "custom"）
# 返回：布尔值，表示 patch 是否成功应用
def apply_custom_patches(custom_patches: List[CustomPatch], testbed:str, patch_type: str = "custom"
) -> bool:
    """
    Apply custom patches to task environment and return a git patch

    Args:
        custom_patches (list): list of patches to apply
        patch_type (str): Type of patch (e.g. "eval", "test")
    Returns:
        bool: whether the patch applied successfully
    """
    if not custom_patches:
        print(f"Patch is empty (patch type {patch_type})")
        return False

    # sort by start line number
    custom_patches = sorted(custom_patches, key=lambda x: x.rough_line_number)
    # split patches by file
    patches_by_file: Dict[str, List[CustomPatch]] = defaultdict(list)
    for patch in custom_patches:
        patches_by_file[patch.file_name].append(patch)


    # Apply patch to testbed directory
    # keep track of line number mapping for each file
    for file_name, patches in patches_by_file.items():
        file_name = os.path.join(testbed, file_name)
        try:
            with open(file_name, "r") as f:
                file = f.read()
        except FileNotFoundError:
            # Patch file not found
            # could be because this is a new file
            file = ""
        try:
            file_ast = ast.parse(file)
        except SyntaxError:
            print(f" Syntax error in file: {file_name}")
            return False

        for patch in patches:
            patch_joined = "\n".join(patch.changed_lines)
            patch_joined = textwrap.dedent(patch_joined)
            try:
                patch_ast = ast.parse(patch_joined)
            except SyntaxError:
                print(f"Syntax error in patch: {file_name}")
                return False
            if patch.patch_type == "rewrite":
                patch_ast = patch_ast.body[0]
                if not (isinstance(patch_ast, ast.FunctionDef) or isinstance(patch_ast, ast.ClassDef)):
                    print(f"Invalid patch: {file_name}")
                    return False
                transformer = ReplaceFunctionTransformer(patch_ast, 0)
                transformer.visit(file_ast)
                if not transformer.any_change_applied:
                    print(f"No change applied: {file_name}")
                    return False
            elif patch.patch_type == "insert":
                file_ast.body.extend(patch_ast.body)
                pathlib.Path(file_name).parent.mkdir(parents=True, exist_ok=True)

        with open(file_name, "w") as f:
            f.write(ast.unparse(file_ast))

    # Patch apply succeeded
    print(f"Custom patch successful (for {file_name} and patch type {patch_type})")
    return True


# 函数：overlap_score
# 功能：计算两个字符串列表之间的匹配得分，基于编辑距离计算字符串相似性
# 参数：a, b - 两个字符串列表（通常为上下文行）
# 返回：匹配得分（数值）
def overlap_score(a: List[str], b: List[str]):
    score = 0
    for j, context_line in enumerate(a):
        if j >= len(b):
            continue
        distance = editdistance.eval(b[j], context_line)
        score += (1 - (distance / max(len(b[j]), len(context_line)))) if len(b[j]) > 0 or len(context_line) > 0 else 0
    return score

# 函数：write_diff_and_reset
# 功能：将当前测试路径下的 git diff 写入文件，并重置仓库至指定提交状态
# 参数：
#   testbed - 工作目录路径（git 仓库所在目录）
#   reference_commit - 用于对比的参考提交
#   target_file - 保存 diff 输出的文件路径
def write_diff_and_reset(testbed: str, reference_commit: str ='', target_file:str = "./processed_patch.diff"):
    repo = git.Repo(testbed)
    repo.git.add(all=True)
    diff = repo.git.diff(reference_commit)
    with open(target_file, "w") as f:
        f.write(diff)

    repo.git.reset('--hard', reference_commit)

# 函数：apply_patch
# 功能：使用 git apply 命令将给定 patch 应用到测试环境目录中
# 参数：
#   patch - diff patch 文本
#   testbed - 测试环境目录路径
# 返回：布尔值，表示 patch 是否成功应用
def apply_patch(patch, testbed):
    repo = git.Repo(testbed)
    with NamedTemporaryFile("w", suffix=".diff") as f:
        f.write(patch)
        try:
            repo.git.apply("-v", f.name)
            return True
        except git.exc.GitCommandError as e:
            return False

# 函数：run
# 功能：根据模型输出文件及指定 patch 类型，选择合适的 patch 应用方式，并最终写入 diff 文件后重置仓库
# 参数：
#   model_output_file - 原始模型输出的文件路径（包含 diff 文本）
#   testbed - 测试环境路径
#   patch_type - patch 类型列表（如 "vanilla", "fuzzy", "custom"）
#   reference_commit - 用于生成 diff 的参考提交标识
#   target_file - 输出 diff 文件路径
def run(model_output_file: str, testbed:str, patch_type: List[str], reference_commit: str, target_file: str):
    with open(model_output_file, "r") as f:
        raw_model_output = f.read()

    success = False
    for patch_type in patch_type:
        if success:
            break
        if patch_type == "fuzzy":
            model_patch = extract_fuzzy_patch(raw_model_output)
            success = apply_fuzzy_patches(fuzzy_patch=model_patch, testbed=testbed)
        elif patch_type == "custom":
            model_patch = extract_custom_patches(raw_model_output)
            success = apply_custom_patches(custom_patches=model_patch, testbed=testbed)
        elif patch_type == "vanilla":
            model_patch = extract_minimal_patch(raw_model_output)
            success = apply_patch(patch=model_patch, testbed=testbed)
        else:
            assert False, f"Unkown patch type {patch_type}"

    write_diff_and_reset(testbed, reference_commit, target_file)

# 函数：main
# 功能：程序入口，解析命令行参数并调用 run 函数启动 patch 处理流程
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_file", default="/root/raw_model_patch.txt", type=str, help="Path to raw model output")
    parser.add_argument("--testbed", default="/testbed/", type=str, help="Path to raw model output")
    parser.add_argument("--patch_type", nargs="+", choices=["vanilla", "fuzzy", "custom"], type=str, help="Type of patch to be extracted")
    parser.add_argument("--reference_commit", required=True, type=str, help="Type of patch to be extracted")
    parser.add_argument("--target_file", default="/root/extracted_patch.diff", type=str, help="Path to raw model output")

    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()


