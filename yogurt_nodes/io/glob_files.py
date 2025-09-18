import os
from itertools import chain
from pathlib import Path
from typing import List, Tuple

from PIL import Image


class GlobFiles:
    """
    使用 glob 模式遍历文件夹，返回匹配的文件路径列表。

    支持在 `glob_pattern` 中按行提供多个模式；结果会依次拼接。
    可选返回相对于 root_directory 的路径。
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
                        "multiline": True,
                        "tooltip": "One glob pattern per line, e.g. '*.txt' or '**/*.py'. Empty lines are ignored.",
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
                "extension_list": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Extension list, multiple extensions separated by commas, e.g. 'txt,py,json' etc. Case insensitive. If empty, all files will be returned.",
                    },
                ),
                "extension_is_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, the extension list will be treated as image extensions.",
                    },
                ),
                "relative": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Return paths relative to the root directory.",
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
        extension_list: str = "",
        extension_is_image: bool = False,
        relative: bool = False,
    ) -> Tuple[List[str]]:
        if not root_directory or not root_directory.strip():
            root_directory = "."

        pattern_lines = [line.strip() for line in glob_pattern.splitlines() if line.strip()]
        if not pattern_lines:
            pattern_lines = ["*"]

        root_path = Path(root_directory)

        if not root_path.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root_directory}")

        if not root_path.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {root_directory}")

        def iter_matches(pattern: str):
            try:
                iterator = root_path.glob(pattern) if glob_mode == "glob" else root_path.rglob(pattern)
                for match in iterator:
                    yield match
            except Exception as exc:
                raise ValueError(f"Invalid glob pattern '{pattern}': {exc}") from exc

        paths_iter = chain.from_iterable(iter_matches(pattern) for pattern in pattern_lines)

        if files_only:
            paths_iter = (p for p in paths_iter if p.is_file())

        if prefix_list:
            prefixes = [item.strip().lower() for item in prefix_list.split(",") if item.strip()]
            if prefixes:
                paths_iter = (
                    p for p in paths_iter if any(p.name.lower().startswith(prefix) for prefix in prefixes)
                )

        if suffix_list:
            suffixes = [item.strip().lower() for item in suffix_list.split(",") if item.strip()]
            if suffixes:
                paths_iter = (
                    p for p in paths_iter if any(p.name.lower().endswith(suffix) for suffix in suffixes)
                )

        if extension_list:
            extensions = [item.strip().lower().lstrip(".") for item in extension_list.split(",") if item.strip()]
            if extensions:
                paths_iter = (
                    p for p in paths_iter if p.suffix.lower().lstrip(".") in extensions
                )

        if extension_is_image:
            image_extensions = {
                ext.lower().lstrip(".") for ext in Image.registered_extensions().keys()
            }
            paths_iter = (
                p for p in paths_iter if p.suffix.lower().lstrip(".") in image_extensions
            )

        root_reference = root_path
        if resolve_path:
            paths_iter = (p.resolve() for p in paths_iter)
            root_reference = root_path.resolve()
        elif full_path:
            paths_iter = (p.absolute() for p in paths_iter)
            root_reference = root_path.absolute()

        if relative:
            def to_relative(path: Path) -> Path:
                try:
                    return path.relative_to(root_reference)
                except ValueError:
                    return Path(os.path.relpath(path, start=os.fspath(root_reference)))

            paths_iter = (to_relative(p) for p in paths_iter)

        if as_posix:
            string_iter = (p.as_posix() for p in paths_iter)
        else:
            string_iter = (str(p) for p in paths_iter)

        if sort_files:
            string_paths = sorted(string_iter, reverse=sort_reverse)
        else:
            string_paths = list(string_iter)

        return (string_paths,)
