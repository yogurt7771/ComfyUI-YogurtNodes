from itertools import cycle, islice, zip_longest
from typing import Any, Callable, Iterable, List, Optional

from ..utils import ANY_TYPE


class ListBinaryOps:
    """ListBinaryOps node.

    Perform union, intersection, difference, zip and related operations on two lists.
    """

    OPERATIONS = (
        "union",
        "union_sorted",
        "intersection",
        "intersection_unique",
        "difference",
        "difference_unique",
        "symmetric_difference",
        "cartesian_product",
        "zip",
        "zip_repeat",
        "zip_concat",
        "zip_longest",
        "zip_longest_concat",
        "interleave",
        "splice",
        "swap_ranges",
        "elementwise_add",
        "elementwise_sub",
        "elementwise_mul",
        "merge_dicts",
        "merge_with_key",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (
                    cls.OPERATIONS,
                    {
                        "default": "union",
                        "tooltip": "Binary operation to apply to the two inputs.",
                    },
                ),
                "list_a": (
                    ANY_TYPE,
                    {"tooltip": "First list-like input."},
                ),
                "list_b": (
                    ANY_TYPE,
                    {"tooltip": "Second list-like input."},
                ),
            },
            "optional": {
                "fill_value": (
                    ANY_TYPE,
                    {
                        "default": None,
                        "tooltip": "Padding value for zip_longest variants and interleave.",
                    },
                ),
                "concat": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Connector inserted between paired elements for concat variants.",
                    },
                ),
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Dictionary key used by merge_with_key.",
                    },
                ),
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": -2**31,
                        "max": 2**31 - 1,
                        "tooltip": "Start index used by splice and swap_ranges.",
                    },
                ),
                "end_index": (
                    "INT",
                    {
                        "default": -1,
                        "min": -2**31,
                        "max": 2**31 - 1,
                        "tooltip": "Exclusive end index for splice. Use -1 to insert only.",
                    },
                ),
                "other_start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": -2**31,
                        "max": 2**31 - 1,
                        "tooltip": "Start index in list_b for swap_ranges.",
                    },
                ),
                "range_length": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "tooltip": "Range length for swap_ranges. -1 auto-matches shared space.",
                    },
                ),
                "sort_reverse": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Reverse sorting for union_sorted.",
                    },
                ),
            },
        }

    RETURN_TYPES = (ANY_TYPE, "INT")
    RETURN_NAMES = ("result", "count")
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListBinaryOps"
    DESCRIPTION = "Perform union, intersection, difference, zip and related operations on two lists."

    def execute(
        self,
        operation: str,
        list_a: Iterable[Any],
        list_b: Iterable[Any],
        fill_value=None,
        concat: str = "",
        key: str = "",
        start_index: int = 0,
        end_index: int = -1,
        other_start_index: int = 0,
        range_length: int = -1,
        sort_reverse: bool = False,
    ):
        seq_a = self._ensure_iterable(list_a, "list_a")
        seq_b = self._ensure_iterable(list_b, "list_b")

        if operation == "union":
            result = self._union(seq_a, seq_b)
        elif operation == "union_sorted":
            result = self._union_sorted(seq_a, seq_b, sort_reverse)
        elif operation == "intersection":
            result = self._intersection(seq_a, seq_b)
        elif operation == "intersection_unique":
            result = self._intersection_unique(seq_a, seq_b)
        elif operation == "difference":
            result = self._difference(seq_a, seq_b)
        elif operation == "difference_unique":
            result = self._difference_unique(seq_a, seq_b)
        elif operation == "symmetric_difference":
            result = self._symmetric_difference(seq_a, seq_b)
        elif operation == "cartesian_product":
            result = self._cartesian_product(seq_a, seq_b)
        elif operation == "zip":
            result = list(zip(seq_a, seq_b))
        elif operation == "zip_repeat":
            result = self._zip_repeat(seq_a, seq_b)
        elif operation == "zip_concat":
            result = self._zip_concat(seq_a, seq_b, concat)
        elif operation == "zip_longest":
            result = list(zip_longest(seq_a, seq_b, fillvalue=fill_value))
        elif operation == "zip_longest_concat":
            result = self._zip_longest_concat(seq_a, seq_b, fill_value, concat)
        elif operation == "interleave":
            result = self._interleave(seq_a, seq_b, fill_value)
        elif operation == "splice":
            result = self._splice(seq_a, seq_b, start_index, end_index)
        elif operation == "swap_ranges":
            result = self._swap_ranges(seq_a, seq_b, start_index, other_start_index, range_length)
        elif operation == "elementwise_add":
            result = self._elementwise_op(seq_a, seq_b, lambda a, b: a + b)
        elif operation == "elementwise_sub":
            result = self._elementwise_op(seq_a, seq_b, lambda a, b: a - b)
        elif operation == "elementwise_mul":
            result = self._elementwise_op(seq_a, seq_b, lambda a, b: a * b)
        elif operation == "merge_dicts":
            result = self._merge_dicts(seq_a, seq_b)
        elif operation == "merge_with_key":
            result = self._merge_with_key(seq_a, seq_b, key)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return (result, len(result))

    @staticmethod
    def _ensure_iterable(data: Iterable[Any], name: str) -> List[Any]:
        if data is None:
            raise ValueError(f"{name} must be an iterable, got None.")
        try:
            return list(data)
        except TypeError as exc:
            raise TypeError(f"{name} must be an iterable.") from exc

    @staticmethod
    def _union(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        result: List[Any] = []
        for item in seq_a:
            if not ListBinaryOps._contains(result, item):
                result.append(item)
        for item in seq_b:
            if not ListBinaryOps._contains(result, item):
                result.append(item)
        return result

    @staticmethod
    def _union_sorted(seq_a: List[Any], seq_b: List[Any], sort_reverse: bool) -> List[Any]:
        unique = ListBinaryOps._union(seq_a, seq_b)
        try:
            return sorted(unique, reverse=sort_reverse)
        except TypeError as exc:
            raise TypeError("Elements are not sortable for union_sorted.") from exc

    @staticmethod
    def _intersection(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        multiset_b = ListBinaryOps._build_multiset(seq_b)
        result: List[Any] = []
        for item in seq_a:
            if ListBinaryOps._consume(multiset_b, item):
                result.append(item)
        return result

    @staticmethod
    def _intersection_unique(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        intersection = ListBinaryOps._intersection(seq_a, seq_b)
        result: List[Any] = []
        for item in intersection:
            if not ListBinaryOps._contains(result, item):
                result.append(item)
        return result

    @staticmethod
    def _difference(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        multiset_b = ListBinaryOps._build_multiset(seq_b)
        result: List[Any] = []
        for item in seq_a:
            if not ListBinaryOps._consume(multiset_b, item):
                result.append(item)
        return result

    @staticmethod
    def _difference_unique(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        difference = ListBinaryOps._difference(seq_a, seq_b)
        result: List[Any] = []
        for item in difference:
            if not ListBinaryOps._contains(result, item):
                result.append(item)
        return result

    @staticmethod
    def _symmetric_difference(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        diff_ab = ListBinaryOps._difference(seq_a, seq_b)
        diff_ba = ListBinaryOps._difference(seq_b, seq_a)
        return diff_ab + diff_ba

    @staticmethod
    def _cartesian_product(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        result: List[Any] = []
        for item_a in seq_a:
            for item_b in seq_b:
                result.append((item_a, item_b))
        return result

    @staticmethod
    def _zip_repeat(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        len_a = len(seq_a)
        len_b = len(seq_b)
        if len_a == 0 or len_b == 0:
            return []
        max_len = max(len_a, len_b)
        iter_a = seq_a if len_a == max_len else list(islice(cycle(seq_a), max_len))
        iter_b = seq_b if len_b == max_len else list(islice(cycle(seq_b), max_len))
        return list(zip(iter_a, iter_b))

    @staticmethod
    def _contains(container: List[Any], item: Any) -> bool:
        for existing in container:
            if existing == item:
                return True
        return False

    @staticmethod
    def _zip_concat(seq_a: List[Any], seq_b: List[Any], concat: str) -> List[str]:
        concat_str = concat if isinstance(concat, str) else str(concat)
        result: List[str] = []
        for value_a, value_b in zip(seq_a, seq_b):
            result.append(f"{value_a}{concat_str}{value_b}")
        return result

    @staticmethod
    def _zip_longest_concat(
        seq_a: List[Any],
        seq_b: List[Any],
        fill_value: Any,
        concat: str,
    ) -> List[str]:
        concat_str = concat if isinstance(concat, str) else str(concat)
        result: List[str] = []
        for value_a, value_b in zip_longest(seq_a, seq_b, fillvalue=fill_value):
            result.append(f"{value_a}{concat_str}{value_b}")
        return result

    @staticmethod
    def _interleave(
        seq_a: List[Any],
        seq_b: List[Any],
        fill_value: Optional[Any],
    ) -> List[Any]:
        sentinel = object()
        result: List[Any] = []
        for item_a, item_b in zip_longest(seq_a, seq_b, fillvalue=sentinel):
            if item_a is not sentinel:
                result.append(item_a)
            elif fill_value is not None:
                result.append(fill_value)
            if item_b is not sentinel:
                result.append(item_b)
            elif fill_value is not None:
                result.append(fill_value)
        return result

    @staticmethod
    def _splice(
        seq_a: List[Any],
        seq_b: List[Any],
        start_index: int,
        end_index: int,
    ) -> List[Any]:
        result = list(seq_a)
        start = ListBinaryOps._normalize_index(start_index, len(result))
        if end_index == -1:
            end = start
        else:
            end = ListBinaryOps._normalize_index(end_index, len(result))
        if start > end:
            start, end = end, start
        return result[:start] + list(seq_b) + result[end:]

    @staticmethod
    def _swap_ranges(
        seq_a: List[Any],
        seq_b: List[Any],
        start_index: int,
        other_start_index: int,
        range_length: int,
    ) -> List[Any]:
        if not seq_a:
            return []
        if not seq_b:
            return list(seq_a)
        len_a = len(seq_a)
        len_b = len(seq_b)
        start_a = ListBinaryOps._normalize_index(start_index, len_a)
        start_b = ListBinaryOps._normalize_index(other_start_index, len_b)
        if range_length < 0:
            range_length = min(len_a - start_a, len_b - start_b)
        end_a = min(start_a + range_length, len_a)
        end_b = min(start_b + range_length, len_b)
        slice_a = seq_a[start_a:end_a]
        slice_b = seq_b[start_b:end_b]
        if len(slice_a) != len(slice_b):
            raise ValueError("swap_ranges requires slices of equal length. Adjust range_length.")
        result = list(seq_a)
        result[start_a:end_a] = slice_b
        return result

    @staticmethod
    def _elementwise_op(
        seq_a: List[Any],
        seq_b: List[Any],
        operator: Callable[[Any, Any], Any],
    ) -> List[Any]:
        result: List[Any] = []
        for item_a, item_b in zip(seq_a, seq_b):
            result.append(operator(item_a, item_b))
        return result

    @staticmethod
    def _merge_dicts(seq_a: List[Any], seq_b: List[Any]) -> List[Any]:
        result: List[Any] = []
        for dict_a, dict_b in zip(seq_a, seq_b):
            if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
                raise TypeError("merge_dicts expects dictionary inputs.")
            merged = dict(dict_a)
            merged.update(dict_b)
            result.append(merged)
        return result

    @staticmethod
    def _merge_with_key(seq_a: List[Any], seq_b: List[Any], key: str) -> List[Any]:
        if not key:
            raise ValueError("merge_with_key requires a non-empty key.")
        lookup = {}
        for item in seq_b:
            if not isinstance(item, dict):
                raise TypeError("merge_with_key expects dictionaries in list_b.")
            if key not in item:
                raise KeyError(f"Key '{key}' missing in list_b item {item}.")
            lookup[item[key]] = item
        result: List[Any] = []
        for item in seq_a:
            if not isinstance(item, dict):
                raise TypeError("merge_with_key expects dictionaries in list_a.")
            if key not in item:
                raise KeyError(f"Key '{key}' missing in list_a item {item}.")
            merged = dict(item)
            other = lookup.get(item[key])
            if other:
                merged.update(other)
            result.append(merged)
        return result

    @staticmethod
    def _build_multiset(values: List[Any]) -> List[List[Any]]:
        multiset: List[List[Any]] = []
        for value in values:
            for entry in multiset:
                if entry[0] == value:
                    entry[1] += 1
                    break
            else:
                multiset.append([value, 1])
        return multiset

    @staticmethod
    def _consume(multiset: List[List[Any]], value: Any) -> bool:
        for entry in multiset:
            if entry[0] == value and entry[1] > 0:
                entry[1] -= 1
                return True
        return False

    @staticmethod
    def _normalize_index(index: int, length: int) -> int:
        if index < 0:
            index = length + index
        return max(0, min(index, length))
