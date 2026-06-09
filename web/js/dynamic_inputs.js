import { app } from "../../../scripts/app.js";

const NODE_CONFIGS = {
    YogurtBatchImages: { prefix: "images", type: "IMAGE", startIndex: 1, minCount: 1, maxCount: 32 },
    YogurtPackAny: { prefix: "item", type: "*", startIndex: 1, minCount: 1, maxCount: 32 },
    YogurtEndNode: { prefix: "data", type: "*", startIndex: 1, minCount: 1, maxCount: 32 },
    YogurtAnyBridge: { prefix: "blackhole", type: "*", startIndex: 1, minCount: 1, maxCount: 32 },
    YogurtStringJoin: { prefix: "item", type: "*", startIndex: 1, minCount: 1, maxCount: 32 },
    YogurtStringFormat: { prefix: "input", type: "*", startIndex: 0, minCount: 1, maxCount: 32 },
    YogurtListConcat: { prefix: "list", type: "*", startIndex: 1, minCount: 2, maxCount: 32 },
    YogurtDictMerge: { prefix: "dict", type: "*", startIndex: 1, minCount: 2, maxCount: 32 },
    YogurtJsonMerge: { prefix: "json", type: "*", startIndex: 1, minCount: 2, maxCount: 32 },
};

function dynamicIndex(config, name) {
    if (typeof name !== "string" || !name.startsWith(config.prefix)) {
        return null;
    }

    const suffix = name.slice(config.prefix.length);
    if (!/^\d+$/.test(suffix)) {
        return null;
    }

    const index = Number(suffix);
    if (index < config.startIndex || index >= config.startIndex + config.maxCount) {
        return null;
    }
    return index;
}

function inputName(config, index) {
    return `${config.prefix}${index}`;
}

function findInputSlot(node, name) {
    return (node.inputs || []).findIndex((input) => input.name === name);
}

function linkedType(node, input) {
    const link = input?.link != null ? node.graph?.links?.[input.link] : null;
    return link?.type || configType(input);
}

function configType(input) {
    return input?._yogurtDynamicType || input?.type || "*";
}

function desiredInputCount(node, config) {
    let count = config.minCount;
    for (const input of node.inputs || []) {
        const index = dynamicIndex(config, input.name);
        if (index != null && input.link != null) {
            const relativeIndex = index - config.startIndex;
            count = Math.max(count, relativeIndex + 2);
        }
    }
    return Math.min(count, config.maxCount);
}

function ensureInput(node, config, index) {
    const name = inputName(config, index);
    let slot = findInputSlot(node, name);
    if (slot === -1) {
        node.addInput(name, config.type);
        slot = findInputSlot(node, name);
    }

    const input = node.inputs?.[slot];
    if (input) {
        input.name = name;
        input.label = name;
        input._yogurtDynamicType = config.type;
        input.type = config.type;
    }
}

function trimInputs(node, config, count) {
    for (let slot = (node.inputs || []).length - 1; slot >= 0; slot--) {
        const input = node.inputs[slot];
        const index = dynamicIndex(config, input.name);
        if (index == null) {
            continue;
        }

        const relativeIndex = index - config.startIndex;
        if (relativeIndex >= count && input.link == null) {
            node.removeInput(slot);
        }
    }
}

function sortDynamicInputs(node, config) {
    const inputs = node.inputs || [];
    const dynamicInputs = inputs
        .filter((input) => dynamicIndex(config, input.name) != null)
        .sort((left, right) => dynamicIndex(config, left.name) - dynamicIndex(config, right.name));
    let dynamicSlot = 0;
    node.inputs = inputs.map((input) =>
        dynamicIndex(config, input.name) == null ? input : dynamicInputs[dynamicSlot++]
    );
}

function syncLinkTargetSlots(node) {
    for (let slot = 0; slot < (node.inputs || []).length; slot++) {
        const input = node.inputs[slot];
        const link = input?.link != null ? node.graph?.links?.[input.link] : null;
        if (link) {
            link.target_slot = slot;
        }
    }
}

function syncInputTypes(node, config) {
    for (const input of node.inputs || []) {
        const index = dynamicIndex(config, input.name);
        if (index == null) {
            continue;
        }

        input._yogurtDynamicType = config.type;
        if (config.type === "*") {
            input.type = linkedType(node, input);
        } else {
            input.type = config.type;
        }
    }
}

function refreshDynamicInputs(node, config) {
    const count = desiredInputCount(node, config);
    for (let offset = 0; offset < count; offset++) {
        ensureInput(node, config, config.startIndex + offset);
    }
    trimInputs(node, config, count);
    sortDynamicInputs(node, config);
    syncLinkTargetSlots(node);
    syncInputTypes(node, config);
    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas(true, true);
}

function refreshSoon(node, config) {
    setTimeout(() => refreshDynamicInputs(node, config), 0);
}

app.registerExtension({
    name: "YogurtNodes.DynamicInputs",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData.name];
        if (!config) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            refreshSoon(this, config);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            refreshSoon(this, config);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            refreshSoon(this, config);
            return result;
        };
    },
});
