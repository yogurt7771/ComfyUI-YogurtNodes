from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from sync_node_docstrings import (  # noqa: E402
    _class_has_node_name,
    _extract_string_assignment,
    _is_exported_node_class,
    build_export_policy,
)


SECTION_ORDER = [
    "Image",
    "Number",
    "String",
    "Logic",
    "Models",
    "IO",
    "LLM",
    "Net",
]

SECTION_TITLES = {
    "en": {
        "Image": "Image Processing Nodes",
        "Number": "Number Processing Nodes",
        "String": "String Processing Nodes",
        "Logic": "Logic Processing Nodes",
        "Models": "Model Nodes",
        "IO": "I/O Operation Nodes",
        "LLM": "Language Model Nodes",
        "Net": "Network Nodes",
        "Other": "Other Nodes",
    },
    "zh": {
        "Image": "图像处理节点",
        "Number": "数字处理节点",
        "String": "字符串处理节点",
        "Logic": "逻辑处理节点",
        "Models": "模型节点",
        "IO": "输入/输出操作节点",
        "LLM": "语言模型节点",
        "Net": "网络节点",
        "Other": "其他节点",
    },
}

SUMMARY_HEADERS = {
    "en": ("Group", "Count"),
    "zh": ("分组", "数量"),
}

TABLE_HEADERS = {
    "en": ("Node", "Class ID", "Category", "Description"),
    "zh": ("节点", "Class ID", "分类", "说明"),
}

FALLBACK_CATEGORY_NAMES = {
    "io": "IO",
    "llm": "LLM",
    "net": "Net",
}


@dataclass(frozen=True)
class NodeInfo:
    class_name: str
    node_name: str
    category: str
    description: str
    group: str

    @property
    def node_id(self) -> str:
        public_class_name = (
            self.class_name[:-4] if self.class_name.endswith("__V3") else self.class_name
        )
        return f"Yogurt{public_class_name}"


@dataclass(frozen=True)
class ReadmeTarget:
    path: Path
    language: str
    start_heading: str
    end_heading: str


README_TARGETS = (
    ReadmeTarget(
        path=Path("README.md"),
        language="en",
        start_heading="## 🔧 Available Nodes",
        end_heading="## 🔑 Gemini API Key Setup",
    ),
    ReadmeTarget(
        path=Path("README_zh.md"),
        language="zh",
        start_heading="## 🔧 可用节点",
        end_heading="## 🔑 Gemini API Key 配置说明",
    ),
)


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.replace("\r\n", "\n").split()).strip()


def _extract_description(node_name: str, docstring: str | None, fallback: str | None) -> str:
    fallback_text = _clean_text(fallback)
    if not docstring:
        return fallback_text

    doc_lines = [line.strip() for line in docstring.splitlines() if line.strip()]
    if not doc_lines:
        return fallback_text

    first_line = doc_lines[0]
    remainder = _clean_text(" ".join(doc_lines[1:]))
    title_line = f"{node_name} node.".lower()
    first_line_normalized = first_line.lower()

    if first_line_normalized == title_line:
        return remainder or fallback_text or first_line

    full_doc = _clean_text(" ".join(doc_lines))
    return full_doc or fallback_text


def _group_from_category(category: str) -> str:
    parts = [part for part in category.split("/") if part]
    if len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[0]
    return "Other"


def _fallback_category(package_name: str) -> str:
    suffix = FALLBACK_CATEGORY_NAMES.get(package_name, package_name.title())
    return f"YogurtNodes/{suffix}"


def collect_exported_nodes(package_root: Path) -> list[NodeInfo]:
    export_policy = build_export_policy(package_root)
    nodes: list[NodeInfo] = []

    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue

        package_name = path.parent.name
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for class_def in tree.body:
            if not isinstance(class_def, ast.ClassDef):
                continue
            if not _is_exported_node_class(class_def, package_name, export_policy):
                continue
            if not _class_has_node_name(class_def):
                continue

            node_name = _extract_string_assignment(class_def, "_NODE_NAME")
            if not node_name:
                continue

            category = _extract_string_assignment(
                class_def,
                "CATEGORY",
            ) or _fallback_category(package_name)
            description = _extract_description(
                node_name,
                ast.get_docstring(class_def, clean=True),
                _extract_string_assignment(class_def, "DESCRIPTION"),
            )

            nodes.append(
                NodeInfo(
                    class_name=class_def.name,
                    node_name=node_name,
                    category=category,
                    description=description,
                    group=_group_from_category(category),
                )
            )

    return nodes


def _section_sort_key(group: str) -> tuple[int, str]:
    if group in SECTION_ORDER:
        return SECTION_ORDER.index(group), group
    return len(SECTION_ORDER), group


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def _render_summary(nodes: list[NodeInfo], language: str) -> list[str]:
    group_header, count_header = SUMMARY_HEADERS[language]
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.group] = counts.get(node.group, 0) + 1

    lines = [
        (
            "This section is auto-generated from exported node classes and their docstrings. "
            "Run `python tools/generate_readme.py` to refresh."
            if language == "en"
            else "这一节由导出节点类及其文档注释自动生成。执行 `python tools/generate_readme.py` 可重新生成。"
        ),
        "",
        (
            f"Total exported nodes: **{len(nodes)}**."
            if language == "en"
            else f"当前导出节点总数：**{len(nodes)}**。"
        ),
        "",
        f"| {group_header} | {count_header} |",
        "| --- | ---: |",
    ]

    for group in sorted(counts, key=_section_sort_key):
        title = SECTION_TITLES[language].get(group, group)
        lines.append(f"| {title} | {counts[group]} |")

    return lines


def _render_group_table(nodes: list[NodeInfo], language: str) -> list[str]:
    node_header, class_header, category_header, description_header = TABLE_HEADERS[language]
    lines = [
        f"| {node_header} | {class_header} | {category_header} | {description_header} |",
        "| --- | --- | --- | --- |",
    ]

    for node in sorted(nodes, key=lambda item: (item.category, item.node_name, item.node_id)):
        lines.append(
            "| "
            f"{_escape_markdown_cell(node.node_name)} | "
            f"`{_escape_markdown_cell(node.node_id)}` | "
            f"`{_escape_markdown_cell(node.category)}` | "
            f"{_escape_markdown_cell(node.description)} |"
        )

    return lines


def render_nodes_section(nodes: list[NodeInfo], language: str) -> str:
    title = "## 🔧 Available Nodes" if language == "en" else "## 🔧 可用节点"
    prefix = (
        'All exported nodes are listed here. ComfyUI display names keep the " (Yogurt Nodes)" suffix at runtime.\n'
        if language == "en"
        else '这里列出当前全部导出节点。ComfyUI 运行时的显示名会自动附加 " (Yogurt Nodes)" 后缀。\n'
    )

    lines = [title, "", prefix]
    lines.extend(_render_summary(nodes, language))
    lines.append("")

    grouped: dict[str, list[NodeInfo]] = {}
    for node in nodes:
        grouped.setdefault(node.group, []).append(node)

    for index, group in enumerate(sorted(grouped, key=_section_sort_key)):
        heading = SECTION_TITLES[language].get(group, group)
        lines.append(f"### {heading}")
        lines.append("")
        lines.extend(_render_group_table(grouped[group], language))
        if index != len(grouped) - 1:
            lines.extend(["", ""])

    return "\n".join(lines).rstrip() + "\n"


def _replace_section(content: str, target: ReadmeTarget, replacement: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(target.start_heading)}\r?\n.*?(?=^{re.escape(target.end_heading)}\r?\n)",
        re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(content):
        raise ValueError(
            f"Could not find section between {target.start_heading!r} and {target.end_heading!r} in {target.path}"
        )
    return pattern.sub(replacement.rstrip() + "\n\n", content, count=1)


def update_readmes(repo_root: Path, package_root: Path, *, write: bool) -> list[Path]:
    nodes = collect_exported_nodes(package_root)
    updated: list[Path] = []

    for target in README_TARGETS:
        path = repo_root / target.path
        content = path.read_text(encoding="utf-8")
        replacement = render_nodes_section(nodes, target.language)
        new_content = _replace_section(content, target, replacement)
        if new_content == content:
            continue
        if write:
            path.write_text(new_content, encoding="utf-8")
        updated.append(path)

    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate README node sections from exported Yogurt node classes.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the repository root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report README files that need updates without writing them.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    package_root = repo_root / "yogurt_nodes"
    if not package_root.exists():
        parser.error(f"package root does not exist: {package_root}")

    updated = update_readmes(repo_root, package_root, write=not args.check)

    if not updated:
        print("README files are up to date.")
        return 0

    action = "Would update" if args.check else "Updated"
    for path in updated:
        print(f"{action} {path.relative_to(repo_root)}")

    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
