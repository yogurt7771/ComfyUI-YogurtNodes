from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportPolicy:
    explicit_exports: dict[str, set[str]]
    star_exported_packages: set[str]


@dataclass(frozen=True)
class FileChange:
    path: Path
    classes: tuple[str, ...]


def _iter_node_packages(package_root: Path):
    for path in sorted(package_root.iterdir()):
        if path.is_dir() and (path / "__init__.py").exists():
            yield path


def build_export_policy(package_root: Path) -> ExportPolicy:
    explicit_exports: dict[str, set[str]] = defaultdict(set)
    star_exported_packages: set[str] = set()

    for package_dir in _iter_node_packages(package_root):
        tree = ast.parse((package_dir / "__init__.py").read_text(encoding="utf-8"))
        package_name = package_dir.name
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if alias.name == "*":
                    star_exported_packages.add(package_name)
                else:
                    explicit_exports[package_name].add(alias.asname or alias.name)

    return ExportPolicy(
        explicit_exports=dict(explicit_exports),
        star_exported_packages=star_exported_packages,
    )


def _class_has_node_name(class_def: ast.ClassDef) -> bool:
    return any(
        isinstance(stmt, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_NODE_NAME"
            for target in stmt.targets
        )
        for stmt in class_def.body
    )


def _extract_string_assignment(
    class_def: ast.ClassDef,
    attribute_name: str,
) -> str | None:
    for stmt in class_def.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == attribute_name
            for target in stmt.targets
        ):
            continue
        try:
            value = ast.literal_eval(stmt.value)
        except Exception:
            return None
        return value if isinstance(value, str) else None
    return None


def _clean_text(value: str) -> str:
    return value.replace('"""', "'''").strip()


def build_docstring_lines(
    node_name: str | None,
    description: str | None,
    indent: str,
) -> list[str]:
    title = _clean_text(node_name or "Node")
    description_text = _clean_text(description or "")

    normalized_title = title.rstrip(".").strip().lower()
    normalized_description = description_text.rstrip(".").strip().lower()

    if description_text and normalized_description not in {
        normalized_title,
        f"{normalized_title} node",
    }:
        return [
            f'{indent}"""{title} node.\n',
            "\n",
            f"{indent}{description_text}\n",
            f'{indent}"""\n',
        ]
    return [f'{indent}"""{title} node."""\n']


def _docstring_edit_span(
    class_def: ast.ClassDef,
    lines: list[str],
) -> tuple[int, int, str]:
    first_stmt = class_def.body[0]
    first_stmt_line = getattr(first_stmt, "lineno", class_def.lineno + 1)
    decorator_lines = [item.lineno for item in getattr(first_stmt, "decorator_list", [])]
    anchor_line = min([first_stmt_line, *decorator_lines])

    anchor_source = lines[anchor_line - 1]
    indent = anchor_source[: len(anchor_source) - len(anchor_source.lstrip())]

    doc_expr = None
    if isinstance(first_stmt, ast.Expr):
        value = first_stmt.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            doc_expr = first_stmt
        elif isinstance(value, ast.Str):
            doc_expr = first_stmt

    if doc_expr is not None:
        return doc_expr.lineno - 1, doc_expr.end_lineno, indent
    return anchor_line - 1, anchor_line - 1, indent


def _is_exported_node_class(
    class_def: ast.ClassDef,
    package_name: str,
    export_policy: ExportPolicy,
) -> bool:
    if package_name in export_policy.star_exported_packages:
        return True
    return class_def.name in export_policy.explicit_exports.get(package_name, set())


def sync_file_docstrings(
    path: Path,
    package_name: str,
    export_policy: ExportPolicy,
    *,
    write: bool,
) -> tuple[bool, tuple[str, ...]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, list[str], str]] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_exported_node_class(node, package_name, export_policy):
            continue
        if not _class_has_node_name(node):
            continue

        node_name = _extract_string_assignment(node, "_NODE_NAME")
        description = _extract_string_assignment(node, "DESCRIPTION")
        start, end, indent = _docstring_edit_span(node, lines)
        replacement = build_docstring_lines(node_name, description, indent)

        current = lines[start:end]
        if current == replacement:
            continue
        edits.append((start, end, replacement, node.name))

    if not edits:
        return False, ()

    if write:
        for start, end, replacement, _class_name in sorted(edits, key=lambda item: item[0], reverse=True):
            lines[start:end] = replacement
        path.write_text("".join(lines), encoding="utf-8")

    return True, tuple(class_name for *_rest, class_name in edits)


def sync_package_docstrings(
    package_root: Path,
    *,
    write: bool,
) -> list[FileChange]:
    export_policy = build_export_policy(package_root)
    changes: list[FileChange] = []

    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        package_name = path.parent.name
        changed, classes = sync_file_docstrings(
            path,
            package_name,
            export_policy,
            write=write,
        )
        if changed:
            changes.append(FileChange(path=path, classes=classes))

    return changes


def _default_package_root() -> Path:
    return Path(__file__).resolve().parents[1] / "yogurt_nodes"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or normalize class docstrings for exported Yogurt nodes.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=_default_package_root(),
        help="Path to the yogurt_nodes package root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report files that need updates without writing them.",
    )
    args = parser.parse_args(argv)

    package_root = args.package_root.resolve()
    if not package_root.exists():
        parser.error(f"package root does not exist: {package_root}")

    changes = sync_package_docstrings(package_root, write=not args.check)

    repo_root = package_root.parent
    if not changes:
        print("Node docstrings are up to date.")
        return 0

    action = "Would update" if args.check else "Updated"
    for change in changes:
        relative_path = change.path.relative_to(repo_root)
        class_list = ", ".join(change.classes)
        print(f"{action} {relative_path}: {class_list}")

    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
