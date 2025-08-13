import json
import copy


def json_merge(json_obj1, json_obj2):
    if isinstance(json_obj1, str):
        json_obj1 = json.loads(json_obj1)
    if isinstance(json_obj2, str):
        json_obj2 = json.loads(json_obj2)
    json_obj1 = copy.deepcopy(json_obj1)
    json_obj2 = copy.deepcopy(json_obj2)
    if isinstance(json_obj1, dict):
        for key, value in json_obj2.items():
            if key in json_obj1:
                # 如果key在json_obj1中，则递归合并
                json_obj1[key] = json_merge(json_obj1[key], value)
            else:
                # 如果key不在json_obj1中，则直接添加
                json_obj1[key] = value
    elif isinstance(json_obj1, list):
        for i, item in enumerate(json_obj2):
            if i < len(json_obj1):
                # 前n个元素，递归合并
                json_obj1[i] = json_merge(json_obj1[i], item)
            else:
                # 如果json1的第i个元素不在json2中，则直接添加
                json_obj1.append(item)
    else:
        # 如果json_obj1不是dict或list，则直接覆盖掉
        json_obj1 = json_obj2
    return json_obj1


def _unescape_string(s):
    """去除字符串中的转义符"""
    result = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] in "$.[\\]":
            # 转义字符，添加被转义的字符
            result.append(s[i + 1])
            i += 2
        else:
            # 普通字符
            result.append(s[i])
            i += 1
    return "".join(result)


def _parse_path(path):
    """
    修正后的路径解析，基于用户的思路但修复了问题
    """
    if not path:
        return []

    # 移除开头的 $ 和 .
    if path.startswith("$"):
        path = path[1:]
    if path.startswith("."):
        path = path[1:]

    if not path:
        return []

    result = []
    i = 0
    t_start = 0

    while i < len(path):  # 使用 while 循环，这样可以控制 i 的增减
        if path[i] == "\\" and i + 1 < len(path) and path[i + 1] in "$.[\\]":
            # 跳过转义符和被转义的字符
            i += 2
            continue

        if path[i] == ".":
            # 点号作为分隔符
            if t_start < i:
                segment = path[t_start:i]
                result.append(_unescape_string(segment))
            i += 1
            t_start = i
            continue

        if path[i] == "[":
            # 开始解析数组索引
            j = i + 1
            while j < len(path):
                if path[j] == "\\" and j + 1 < len(path):
                    # 跳过转义字符
                    j += 2
                elif path[j] == "]":
                    # 找到匹配的右括号
                    bracket_content = path[i + 1 : j]
                    bracket_content = _unescape_string(bracket_content)

                    if bracket_content.isdigit():
                        # 是数字索引
                        if t_start < i:
                            segment = path[t_start:i]
                            result.append(_unescape_string(segment))
                        result.append(int(bracket_content))
                        i = j + 1
                        t_start = i
                        break
                    else:
                        # 不是数字，当作普通字符处理
                        j += 1
                else:
                    j += 1
            else:
                # 没找到匹配的右括号，当作普通字符处理
                i += 1
            continue

        # 普通字符，继续
        i += 1

    # 处理最后一段
    if t_start < len(path):
        segment = path[t_start:]
        result.append(_unescape_string(segment))

    return result


def json_get_path(json_data, path, raise_on_nonexist=False):
    """最终版本的JSON属性获取"""
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    def _get_recursive(data, path_parts):
        if not path_parts:
            return data

        key = path_parts[0]
        remaining = path_parts[1:]

        if isinstance(key, int):
            # 数组索引
            if not isinstance(data, list):
                if raise_on_nonexist:
                    raise KeyError(f"Cannot access array index [{key}] on non-array")
                return None

            if key < 0 or key >= len(data):
                if raise_on_nonexist:
                    raise KeyError(
                        f"Array index [{key}] out of bounds. Array length: {len(data)}"
                    )
                return None

            return _get_recursive(data[key], remaining)
        else:
            # 对象属性
            if not isinstance(data, dict):
                if raise_on_nonexist:
                    raise KeyError(f"Cannot access property '{key}' on non-object")
                return None

            if key not in data:
                if raise_on_nonexist:
                    raise KeyError(f"Property '{key}' not found")
                return None

            return _get_recursive(data[key], remaining)

    path_parts = _parse_path(path)
    return _get_recursive(json_data, path_parts)


def json_set_path(json_data, path, value, raise_on_nonexist=False):
    """最终版本的JSON属性设置"""
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    def _set_recursive(data, path_parts, value):
        if not path_parts:
            return value

        key = path_parts[0]
        remaining = path_parts[1:]

        if isinstance(key, int):
            # 数组索引
            if not isinstance(data, list):
                raise ValueError(f"Cannot access array index [{key}] on non-array")

            # 严格模式下检查数组边界
            if raise_on_nonexist and key > len(data):
                raise IndexError(
                    f"Array index [{key}] beyond current length {len(data)}. Maximum allowed: {len(data)}"
                )

            # 扩展数组
            while len(data) <= key:
                data.append(None)

            if not remaining:
                # 最后一个键，设置值
                data[key] = value
            else:
                # 需要继续深入
                if data[key] is None:
                    # 根据下一个键的类型创建结构
                    next_key = remaining[0]
                    data[key] = [] if isinstance(next_key, int) else {}

                data[key] = _set_recursive(data[key], remaining, value)

            return data
        else:
            # 对象属性
            if not isinstance(data, dict):
                raise ValueError(f"Cannot access property '{key}' on non-object")

            if not remaining:
                # 最后一个键，设置值
                data[key] = value
            else:
                # 需要继续深入
                if key not in data:
                    if raise_on_nonexist:
                        raise KeyError(f"Property '{key}' not found")

                    # 根据下一个键的类型创建结构
                    next_key = remaining[0]
                    data[key] = [] if isinstance(next_key, int) else {}

                data[key] = _set_recursive(data[key], remaining, value)

            return data

    if not path:
        raise ValueError("Cannot set the root object")

    path_parts = _parse_path(path)
    _set_recursive(json_data, path_parts, value)
    return json_data
