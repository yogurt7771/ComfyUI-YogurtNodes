import re


class RegexNode:
    R"""
    Regex Node Toolkit
    ========================

    This node provides powerful regex-based string extraction and replacement for multiline text.

    Parameters:
    -----------
    - text (STRING, multiline):
        The input text to process. Supports multiline content.
    - pattern (STRING, multiline):
        The regular expression pattern. Supports Python regex syntax, including flags like (?s) for DOTALL.
        Use parentheses () to define capture groups for use in format_or_replacement.
    - mode (CHOICE):
        The operation mode:
            - 'extract': extract matches by count (see count param)
            - 'replace': replace matches by count (see count param, count=0 means replace all)
    - format_or_replacement (STRING, multiline):
        In 'extract' mode, this is the output format for each match (supports group refs like \g<1>).
        In 'replace' mode, this is the replacement string (supports group refs).
    - count (INT):
        In 'extract' or 'replace' mode:
            - If positive, operate on the first count matches
            - If negative, operate on the last abs(count) matches
            - If zero or abs(count) exceeds total matches, operate on all
    - joiner (STRING):
        Only used in 'extract' mode. The string used to join multiple extracted results. Default is '\n'.

    Modes:
    ------
    1. extract: Returns selected matches, formatted by format_or_replacement, joined by joiner.
    2. replace: Returns text with the first/last count matches replaced by format_or_replacement. If count=0 or abs(count)>=matches, replaces all.

    Group Reference Examples:
    ------------------------
    - \g<0>: The entire match
    - \g<1>: The first capture group
    - \g<2>: The second capture group
      (Use parentheses in pattern to define groups)

    Typical Use Cases:
    ------------------
    - Extracting JSON blocks, numbers, or custom patterns from text
    - Replacing or removing sensitive information
    - Formatting extracted data using group references
    - Cleaning up logs or markdown content

    Tips:
    -----
    - Use [\s\S] instead of . in pattern to match across lines (or rely on DOTALL mode)
    - For advanced regex, refer to Python's re module documentation

    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "The input text to process. Supports multiline content.",
                    },
                ),
                "pattern": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "The regular expression pattern. Supports Python regex syntax, including flags like (?s) for DOTALL.",
                    },
                ),
                "mode": (
                    ["extract", "replace"],
                    {
                        "default": "extract",
                        "tooltip": "The operation mode: 'extract' or 'replace'. extract keep the matched content as `format_or_replacement`, replace replace the matched content with `format_or_replacement`.",
                    },
                ),
                "format_or_replacement": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "In 'extract' mode, this is the output format for each match (supports group refs like \\g<1>). In 'replace' mode, this is the replacement string (supports group refs).",
                    },
                ),
                "count": (
                    "INT",
                    {
                        "default": 0,
                        "step": 1,
                        "tooltip": "In 'extract' or 'replace' mode: If positive, operate on the first count matches; if negative, operate on the last abs(count) matches; if zero or abs(count) exceeds total matches, operate on all.",
                    },
                ),
                "joiner": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Only used in 'extract' mode. The string used to join multiple extracted results. Default is empty.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = False

    FUNCTION = "execute"

    _NODE_NAME = "Regex Node"
    DESCRIPTION = "Regex-based extraction and replacement for multiline text."
    CATEGORY = "ZnzmoNodes/String"

    def execute(
        self,
        text: str,
        pattern: str,
        mode: str = "extract",
        format_or_replacement: str = "",
        count: int = 0,
        joiner: str = "\n",
    ):
        if mode == "extract":
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
            n = len(matches)
            if n == 0 or count == 0 or abs(count) >= n:
                selected = matches
            elif count > 0:
                selected = matches[:count]
            else:
                selected = matches[count:]
            results = []
            for match in selected:
                if format_or_replacement.strip():
                    try:
                        results.append(match.expand(format_or_replacement))
                    except Exception:
                        results.append(match.group(0))
                else:
                    results.append(match.group(0))
            return (joiner.join(results),)
        elif mode == "replace":
            try:
                matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
                n = len(matches)
                if n == 0 or count == 0 or abs(count) >= n:
                    # 全部替换
                    result = re.sub(
                        pattern,
                        format_or_replacement,
                        text,
                        flags=re.MULTILINE | re.DOTALL,
                    )
                    return (result,)
                if count > 0:
                    to_replace = matches[:count]
                else:
                    to_replace = matches[count:]
                result = []
                last_end = 0
                for m in matches:
                    if m in to_replace:
                        result.append(text[last_end : m.start()])
                        try:
                            result.append(m.expand(format_or_replacement))
                        except Exception:
                            result.append(format_or_replacement)
                    else:
                        result.append(text[last_end : m.end()])
                    last_end = m.end()
                result.append(text[last_end:])
                return ("".join(result),)
            except Exception as e:
                return (f"[Regex Replace Error] {e}",)
        else:
            return ("[Invalid mode]",)
