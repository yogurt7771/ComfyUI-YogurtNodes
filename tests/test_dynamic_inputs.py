import json
import importlib.util
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_package_stubs():
    yogurt_pkg = types.ModuleType("yogurt_nodes")
    yogurt_pkg.__path__ = [str(REPO_ROOT / "yogurt_nodes")]
    sys.modules["yogurt_nodes"] = yogurt_pkg

    for package_name in ("image", "logic", "string", "io"):
        module = types.ModuleType(f"yogurt_nodes.{package_name}")
        module.__path__ = [str(REPO_ROOT / "yogurt_nodes" / package_name)]
        sys.modules[f"yogurt_nodes.{package_name}"] = module

    utils_mod = types.ModuleType("yogurt_nodes.utils")
    utils_mod.__path__ = [str(REPO_ROOT / "yogurt_nodes" / "utils")]
    utils_mod.ANY_TYPE = "*"
    sys.modules["yogurt_nodes.utils"] = utils_mod

    dynamic_inputs = load_module(
        "yogurt_nodes.utils.dynamic_inputs",
        REPO_ROOT / "yogurt_nodes" / "utils" / "dynamic_inputs.py",
    )
    utils_mod.DYNAMIC_INPUT_COUNT = dynamic_inputs.DYNAMIC_INPUT_COUNT
    utils_mod.make_dynamic_inputs = dynamic_inputs.make_dynamic_inputs
    utils_mod.ordered_dynamic_values = dynamic_inputs.ordered_dynamic_values

    def json_merge(left, right):
        merged = dict(left)
        merged.update(right)
        return merged

    utils_mod.json_merge = json_merge
    utils_mod.json_get_path = lambda *_args, **_kwargs: None
    utils_mod.json_set_path = lambda data, *_args, **_kwargs: data


def prepare_comfy_stub():
    comfy_pkg = types.ModuleType("comfy")
    utils_mod = types.ModuleType("comfy.utils")

    def lanczos(samples, width, height):
        return F.interpolate(samples, size=(height, width), mode="bilinear", align_corners=False)

    utils_mod.lanczos = lanczos
    comfy_pkg.utils = utils_mod
    sys.modules["comfy"] = comfy_pkg
    sys.modules["comfy.utils"] = utils_mod


class DynamicInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_package_stubs()
        prepare_comfy_stub()
        cls.batch_module = load_module(
            "yogurt_nodes.image.batch_image",
            REPO_ROOT / "yogurt_nodes" / "image" / "batch_image.py",
        )
        cls.pack_module = load_module(
            "yogurt_nodes.logic.pack_any",
            REPO_ROOT / "yogurt_nodes" / "logic" / "pack_any.py",
        )
        cls.string_join_module = load_module(
            "yogurt_nodes.string.string_join",
            REPO_ROOT / "yogurt_nodes" / "string" / "string_join.py",
        )
        cls.list_advanced_module = load_module(
            "yogurt_nodes.logic.list_advanced",
            REPO_ROOT / "yogurt_nodes" / "logic" / "list_advanced.py",
        )
        cls.dict_advanced_module = load_module(
            "yogurt_nodes.logic.dict_advanced",
            REPO_ROOT / "yogurt_nodes" / "logic" / "dict_advanced.py",
        )
        cls.json_ops_module = load_module(
            "yogurt_nodes.logic.json_ops",
            REPO_ROOT / "yogurt_nodes" / "logic" / "json_ops.py",
        )

    def test_batch_images_reads_dynamic_slots_through_max_count(self):
        image1 = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
        image32 = torch.ones((1, 2, 3, 3), dtype=torch.float32)

        batch, count, width, height, channels = self.batch_module.BatchImages().batch_image(
            interpolation="nearest",
            method="pad",
            pad_value=1.0,
            images1=image1,
            images32=image32,
        )

        self.assertEqual(self.batch_module.IMAGE_COUNT, 32)
        self.assertEqual(count, 2)
        self.assertEqual(tuple(batch.shape), (2, 2, 3, 3))
        self.assertEqual((width, height, channels), (3, 2, 3))
        self.assertTrue(torch.equal(batch[0], image1[0]))
        self.assertTrue(torch.equal(batch[1], image32[0]))

    def test_pack_any_reads_dynamic_items_through_max_count(self):
        pack = self.pack_module.PackAny().execute(
            item1="first",
            item9="ninth",
            item32="thirty-second",
        )[0]

        self.assertEqual(self.pack_module.PACK_NUM, 32)
        self.assertEqual(len(pack), 32)
        self.assertEqual(pack[0], "first")
        self.assertIsNone(pack[1])
        self.assertEqual(pack[8], "ninth")
        self.assertEqual(pack[31], "thirty-second")

    def test_string_join_reads_dynamic_items_in_numeric_order(self):
        result = self.string_join_module.StringJoin().execute(
            "|",
            item1="first",
            item10="tenth",
            item32="thirty-second",
            item2="second",
        )[0]

        self.assertEqual(self.string_join_module.INPUT_COUNT, 32)
        self.assertEqual(result, "first|second|tenth|thirty-second")

    def test_merge_nodes_read_dynamic_extra_inputs(self):
        list_result = self.list_advanced_module.ListConcat().execute(
            [1],
            [2],
            list3=[3],
            list10=[10],
        )[0]
        dict_result = self.dict_advanced_module.DictMerge().execute(
            {"a": 1},
            {"b": 2},
            dict3={"c": 3},
            dict10={"j": 10},
        )[0]
        json_result = self.json_ops_module.JsonMerge().execute(
            {"a": 1},
            {"b": 2},
            json3={"c": 3},
            json10={"j": 10},
        )[0]

        self.assertEqual(list_result, [1, 2, 3, 10])
        self.assertEqual(dict_result, {"a": 1, "b": 2, "c": 3, "j": 10})
        self.assertEqual(json_result, {"a": 1, "b": 2, "c": 3, "j": 10})

    def test_dynamic_input_js_registers_multi_input_nodes(self):
        extension_path = REPO_ROOT / "web" / "js" / "dynamic_inputs.js"

        self.assertTrue(extension_path.exists())
        source = extension_path.read_text(encoding="utf-8")
        self.assertIn("maxCount: 32", source)
        self.assertNotIn("maxCount: 20", source)
        for node_name in (
            "YogurtBatchImages",
            "YogurtPackAny",
            "YogurtEndNode",
            "YogurtAnyBridge",
            "YogurtStringJoin",
            "YogurtStringFormat",
            "YogurtListConcat",
            "YogurtDictMerge",
            "YogurtJsonMerge",
        ):
            self.assertIn(node_name, source)
        self.assertIn("onConnectionsChange", source)

    def test_dynamic_input_js_expands_trims_and_syncs_types(self):
        node_exe = shutil.which("node")
        if node_exe is None:
            self.skipTest("node executable is not available")

        extension_path = REPO_ROOT / "web" / "js" / "dynamic_inputs.js"
        source = extension_path.read_text(encoding="utf-8")
        script = f"""
const assert = require("assert");
const vm = require("vm");
let source = {json.dumps(source)};
source = source.replace(
  'import {{ app }} from "../../../scripts/app.js";',
  'const app = {{ registerExtension(_extension) {{}} }};'
);

source += `
const config = NODE_CONFIGS.YogurtPackAny;
const node = {{
  inputs: [
    {{ name: "item1", type: "*", link: 1 }},
    {{ name: "item2", type: "*", link: null }},
    {{ name: "item9", type: "*", link: 9 }},
    {{ name: "item10", type: "*", link: null }},
    {{ name: "item11", type: "*", link: null }},
    {{ name: "separator", type: "STRING", link: null }},
  ],
  graph: {{
    links: {{
      1: {{ type: "STRING", target_slot: 0 }},
      9: {{ type: "IMAGE", target_slot: 2 }},
    }},
    dirty: false,
    setDirtyCanvas() {{
      this.dirty = true;
    }},
  }},
  addInput(name, type) {{
    this.inputs.push({{ name, type, link: null }});
  }},
  removeInput(slot) {{
    this.inputs.splice(slot, 1);
  }},
  computeSize() {{
    return [320, 240];
  }},
  setSize(size) {{
    this.size = size;
  }},
}};

refreshDynamicInputs(node, config);

const itemNames = node.inputs
  .map((input) => input.name)
  .filter((name) => name.startsWith("item") && Number.isInteger(Number(name.slice(4))));
assert.deepStrictEqual(
  itemNames,
  ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10"]
);
assert.strictEqual(node.inputs.find((input) => input.name === "item1").type, "STRING");
assert.strictEqual(node.inputs.find((input) => input.name === "item9").type, "IMAGE");
assert.strictEqual(node.graph.links[1].target_slot, 0);
assert.strictEqual(
  node.graph.links[9].target_slot,
  node.inputs.findIndex((input) => input.name === "item9")
);
assert.strictEqual(node.inputs.some((input) => input.name === "item11"), false);
assert.strictEqual(node.inputs.some((input) => input.name === "separator"), true);
assert.deepStrictEqual(node.size, [320, 240]);
assert.strictEqual(node.graph.dirty, true);
`;

vm.runInNewContext(source, {{ assert }});
"""

        subprocess.run([node_exe, "-e", script], check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    unittest.main()
