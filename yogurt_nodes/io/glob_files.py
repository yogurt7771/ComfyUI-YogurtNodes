from pathlib import Path
from typing import List, Tuple


class GlobFiles:
    """
    使用 glob 模式遍历文件夹，返回匹配的文件路径列表
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_directory": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Root directory path, from which to start searching for files.",
                    },
                ),
                "glob_pattern": (
                    "STRING",
                    {
                        "default": "*",
                        "tooltip": "Glob search pattern, e.g. '*.txt', '**/*.py', 'test_*.json' etc.",
                    },
                ),
                "sort_files": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Sort the file paths in alphabetical order.",
                    },
                ),
                "sort_reverse": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Sort in descending order."},
                ),
                "glob_mode": (
                    ["glob", "rglob"],
                    {
                        "default": "glob",
                        "tooltip": "glob from left to right, rglob from right to left",
                    },
                ),
                "files_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Return only files, exclude directories.",
                    },
                ),
                "as_posix": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Convert the path to a POSIX format.",
                    },
                ),
                "full_path": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Return an absolute version of this path by prepending the current working directory. No normalization or symlink resolution is performed.",
                    },
                ),
                "resolve_path": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Make the path absolute, resolving all symlinks on the way and also normalizing it.",
                    },
                ),
                "prefix_list": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Prefix list, multiple prefixes separated by commas, e.g. 'pre-1,pre-2,pre-3' etc. Case insensitive. If empty, all files will be returned.",
                    },
                ),
                "suffix_list": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Suffix list, multiple suffixes separated by commas, e.g. 'txt,py,json' etc. Case insensitive. If empty, all files will be returned.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "Glob Files"
    DESCRIPTION = "Use glob pattern to traverse the folder, return the matching file path list"
    CATEGORY = "YogurtNodes/IO"

    def execute(
        self,
        root_directory: str = ".",
        glob_pattern: str = "*",
        sort_files: bool = False,
        sort_reverse: bool = False,
        glob_mode: str = "glob",
        files_only: bool = False,
        as_posix: bool = False,
        full_path: bool = False,
        resolve_path: bool = False,
        prefix_list: str = "",
        suffix_list: str = "",
    ) -> Tuple[List[str]]:
        # 输入验证
        if not root_directory or not root_directory.strip():
            root_directory = "."

        if not glob_pattern or not glob_pattern.strip():
            glob_pattern = "*"

        root_path = Path(root_directory)

        # 检查根目录是否存在
        if not root_path.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root_directory}")

        if not root_path.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {root_directory}")

        # 获取路径迭代器
        try:
            if glob_mode == "glob":
                paths_iterator = root_path.glob(glob_pattern)
            elif glob_mode == "rglob":
                paths_iterator = root_path.rglob(glob_pattern)
            else:
                raise ValueError(
                    f"Invalid glob mode: {glob_mode}. Must be 'glob' or 'rglob'."
                )
        except Exception as e:
            raise ValueError(f"Invalid glob pattern '{glob_pattern}': {e}")

        # 过滤只要文件（如果需要）
        if files_only:
            paths_iterator = (p for p in paths_iterator if p.is_file())

        # 过滤前缀
        if prefix_list:
            prefix_list = prefix_list.split(",")
            paths_iterator = (p for p in paths_iterator if any(p.name.lower().startswith(prefix.lower()) for prefix in prefix_list))

        # 过滤后缀
        if suffix_list:
            suffix_list = suffix_list.split(",")
            paths_iterator = (p for p in paths_iterator if any(p.name.lower().endswith(suffix.lower()) for suffix in suffix_list))

        # 路径转换
        if resolve_path:
            paths_iterator = (p.resolve() for p in paths_iterator)
        elif full_path:
            paths_iterator = (p.absolute() for p in paths_iterator)

        # 字符串转换
        if as_posix:
            string_paths_iterator = (p.as_posix() for p in paths_iterator)
        else:
            string_paths_iterator = (str(p) for p in paths_iterator)

        # 排序和返回
        if sort_files:
            final_paths = sorted(string_paths_iterator, reverse=sort_reverse)
        else:
            final_paths = list(string_paths_iterator)

        return (final_paths,)
