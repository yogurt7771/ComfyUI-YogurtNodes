import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "YogurtNodes.PreviewAnyBridge",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YogurtPreviewAnyBridge") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.showValueWidget = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
                this.showValueWidget.inputEl.readOnly = true;
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message]);
                this.showValueWidget.value = message.text[0];
            };
            console.log("Preview Any Bridge (Yogurt Nodes) registered");
        }
    },
}); 