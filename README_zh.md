# ComfyUI-YogurtNodes

ComfyUI-YogurtNodes是ComfyUI的自定义节点集合，提供一系列实用的图像处理和工作流增强功能。

## ✨ 特点

- 自定义节点集成
- 易于使用的图像处理功能
- 与ComfyUI工作流完全兼容
- 文本和图像处理能力
- 高级字符串处理工具
- 模型管理和选择工具
- 全面的输入/输出操作支持
- 集成Gemini API的语言和图像理解功能
- 集成OpenAI API的文本生成和图像理解功能
- 逻辑控制节点支持复杂工作流

## 📦 安装

### 要求

- ComfyUI（已安装并运行）
- Python 3.x
- 需要的Python包：
  - numpy
  - pillow
  - google-genai (对于Gemini节点)
  - openai (对于OpenAI和OpenRouter节点)
  - requests (对于API调用)
  - opencv-python (对于泊松融合)

### 安装步骤

1. 导航到您的ComfyUI自定义节点目录：

```bash
cd custom_nodes
```

2. 克隆此仓库：

```bash
git clone https://github.com/yogurt7771/ComfyUI-YogurtNodes.git
```

3. 安装依赖：

```bash
cd ComfyUI-YogurtNodes
pip install -r requirements.txt
```

## 🚀 使用方法

1. 启动ComfyUI
2. 在节点浏览器中查找"Yogurt Nodes"类别
3. 将所需节点拖放到您的工作流中

## 🔧 可用节点

这里列出当前全部导出节点。ComfyUI 运行时的显示名会自动附加 " (Yogurt Nodes)" 后缀。

这一节由导出节点类及其文档注释自动生成。执行 `python tools/generate_readme.py` 可重新生成。

当前导出节点总数：**183**。

| 分组 | 数量 |
| --- | ---: |
| 图像处理节点 | 11 |
| 遮罩节点 | 5 |
| 数字处理节点 | 2 |
| 字符串处理节点 | 8 |
| 逻辑处理节点 | 41 |
| 模型节点 | 17 |
| 输入/输出操作节点 | 36 |
| 语言模型节点 | 55 |
| 网络节点 | 8 |

### 图像处理节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Add Text To Image | `YogurtAddTextToImage` | `YogurtNodes/Image` | Add text to image. |
| Batch Images | `YogurtBatchImages` | `YogurtNodes/Image` | Batch images. |
| Get Image Size | `YogurtGetImageSize` | `YogurtNodes/Image` | Get image size information. |
| H/L Frequency Detail Restore Threshold | `YogurtHLFrequencyDetailRestoreThreshold` | `YogurtNodes/Image` | H/L frequency detail restore with independent high/low delta thresholds. |
| Image Crop By Mask | `YogurtImageCropByMask` | `YogurtNodes/Image` | Crop image to the minimum bounding box of the mask above threshold. |
| Image Scale To Total Pixels Advanced | `YogurtImageScaleToTotalPixelsAdvanced` | `YogurtNodes/Image` | Image Scale To Total Pixels Advanced. |
| Image Tile (Seam Mask) | `YogurtImageTileWithSeamMask` | `YogurtNodes/Image` | Split image into overlapped tiles and generate inpaint masks (white=inpaint, black=reference). |
| Image Untile (Seam Mask) | `YogurtImageUntileWithSeamMask` | `YogurtNodes/Image` | Merge overlapped tiles back to one image with seam feathering (mask + overlap-based smooth transition). |
| Poisson Blend | `YogurtPoissonBlend` | `YogurtNodes/Image` | 使用OpenCV泊松融合(seamlessClone)将前景融合到背景。 |
| Replace Image In Batch | `YogurtReplaceImageInBatch` | `YogurtNodes/Image` | Replace one image inside an image batch at the given index. |
| Tile Info To TTP Image Assy Args | `YogurtTileInfoToTTPImageAssyArgs` | `YogurtNodes/Image` | Convert tile_info to TTP_Image_Assy inputs: positions/original_size/grid_size/padding. |


### 遮罩节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Composite Repaint Regions | `YogurtCompositeRepaintRegions` | `YogurtNodes/Masks` | Composite repainted region crops back onto a base image. |
| Crop Image By Regions | `YogurtCropImageByRegions` | `YogurtNodes/Masks` | Crop repaint image and mask batches from dynamic mask regions. |
| Mask Region Planner | `YogurtMaskRegionPlanner` | `YogurtNodes/Masks` | Plan dynamic fixed-size repaint tiles from a mask batch. |
| Preview Repaint Regions | `YogurtPreviewRepaintRegions` | `YogurtNodes/Masks` | Draw planned repaint region boxes over the image. |
| Split Mask | `YogurtSplitMask` | `YogurtNodes/Masks` | Split a combined mask into one mask batch item per connected component. |


### 数字处理节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Range | `YogurtRange` | `YogurtNodes/Number` | get a number from a range |
| RangeItem | `YogurtRangeItem` | `YogurtNodes/Number` | get a value from a range |


### 字符串处理节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Regex Node | `YogurtRegexNode` | `YogurtNodes/String` | Regex-based extraction and replacement for multiline text. |
| Replace Delimiter | `YogurtReplaceDelimiter` | `YogurtNodes/String` | Replace delimiter in string. Support regex |
| String Concat | `YogurtStringConcat` | `YogurtNodes/String` | 拼接多个字符串，支持自定义分隔符和可变数量的输入 |
| String Format | `YogurtStringFormat` | `YogurtNodes/String` | Format strings |
| String Join | `YogurtStringJoin` | `YogurtNodes/String` | 将多个字符串使用指定连接符连接 |
| String Lines Count | `YogurtStringLinesCount` | `YogurtNodes/String` | Get the number of lines in a multiline string |
| String Lines Switch | `YogurtStringLinesSwitch` | `YogurtNodes/String` | Get line from multiline string by index |
| String To Value | `YogurtStringToValue` | `YogurtNodes/String` | Get value from string |


### 逻辑处理节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| DataSize | `YogurtDataSize` | `YogurtNodes/Logic` | Get the size/length of any data structure |
| DictContainsKey | `YogurtDictContainsKey` | `YogurtNodes/Logic` | Check if a dictionary contains a specific key |
| DictContainsValue | `YogurtDictContainsValue` | `YogurtNodes/Logic` | Check if a dictionary contains a specific value |
| DictFilter | `YogurtDictFilter` | `YogurtNodes/Logic` | Filter dictionary entries based on key or value patterns |
| DictFromLists | `YogurtDictFromLists` | `YogurtNodes/Logic` | Create a dictionary from a list of keys and a list of values |
| DictGet | `YogurtDictGet` | `YogurtNodes/Logic` | Get a value by key from any dict-like object |
| DictInvert | `YogurtDictInvert` | `YogurtNodes/Logic` | Invert a dictionary (swap keys and values) |
| DictKeys | `YogurtDictKeys` | `YogurtNodes/Logic` | Get all keys from any dict-like object |
| DictLength | `YogurtDictLength` | `YogurtNodes/Logic` | Get the length (number of keys) of any dict-like object |
| DictMerge | `YogurtDictMerge` | `YogurtNodes/Logic` | Merge multiple dictionaries |
| DictSubset | `YogurtDictSubset` | `YogurtNodes/Logic` | Get a subset of a dict-like object by specifying keys |
| DictValues | `YogurtDictValues` | `YogurtNodes/Logic` | Get all values from any dict-like object |
| EndNode | `YogurtEndNode` | `YogurtNodes/Logic` | End |
| IsEmpty | `YogurtIsEmpty` | `YogurtNodes/Logic` | Check if a data structure is empty |
| JsonDeepCopy | `YogurtJsonDeepCopy` | `YogurtNodes/Logic` | Create a deep copy of JSON object |
| JsonFlatten | `YogurtJsonFlatten` | `YogurtNodes/Logic` | Flatten nested JSON object to flat key-value pairs |
| JsonGetPath | `YogurtJsonGetPath` | `YogurtNodes/Logic` | Get value from JSON object using JSONPath |
| JsonMerge | `YogurtJsonMerge` | `YogurtNodes/Logic` | Merge multiple JSON objects using deep merge |
| JsonParse | `YogurtJsonParse` | `YogurtNodes/Logic` | Parse JSON string to object |
| JsonPathExists | `YogurtJsonPathExists` | `YogurtNodes/Logic` | Check if a path exists in JSON object |
| JsonSetPath | `YogurtJsonSetPath` | `YogurtNodes/Logic` | Set value in JSON object using JSONPath |
| JsonStringify | `YogurtJsonStringify` | `YogurtNodes/Logic` | Convert object to JSON string |
| JsonUnflatten | `YogurtJsonUnflatten` | `YogurtNodes/Logic` | Unflatten flat JSON object back to nested structure |
| JsonValidate | `YogurtJsonValidate` | `YogurtNodes/Logic` | Validate JSON data structure |
| ListBinaryOps | `YogurtListBinaryOps` | `YogurtNodes/Logic` | Perform union, intersection, difference, zip and related operations on two lists. |
| ListConcat | `YogurtListConcat` | `YogurtNodes/Logic` | Concatenate multiple lists |
| ListContains | `YogurtListContains` | `YogurtNodes/Logic` | Check if a list contains a specific element |
| ListFilter | `YogurtListFilter` | `YogurtNodes/Logic` | Filter list elements based on regex pattern |
| ListFind | `YogurtListFind` | `YogurtNodes/Logic` | Find the index of an element in a list |
| ListIndex | `YogurtListIndex` | `YogurtNodes/Logic` | 通过索引从任何列表类型对象中获取元素 |
| ListJoin | `YogurtListJoin` | `YogurtNodes/Logic` | Join list elements into a string |
| ListLength | `YogurtListLength` | `YogurtNodes/Logic` | 获取任何列表类型对象的长度 |
| ListSlice | `YogurtListSlice` | `YogurtNodes/Logic` | 从任何列表类型对象中获取切片 |
| ListUnique | `YogurtListUnique` | `YogurtNodes/Logic` | Remove duplicate elements from a list while preserving order |
| None | `YogurtNoneNode` | `YogurtNodes/Logic` | Return None. |
| PackAny | `YogurtPackAny` | `YogurtNodes/Logic` | Pack any |
| StringSplit | `YogurtStringSplit` | `YogurtNodes/Logic` | Split a string into a list |
| Switch | `YogurtSwitch` | `YogurtNodes/Logic` | Switch |
| ToDict | `YogurtToDict` | `YogurtNodes/Logic` | Convert pairs or mapping to a dictionary |
| ToList | `YogurtToList` | `YogurtNodes/Logic` | Convert any iterable to a list |
| UnpackAny | `YogurtUnpackAny` | `YogurtNodes/Logic` | Unpack any |


### 模型节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Checkpoint Selector | `YogurtCheckpointSelector` | `YogurtNodes/Models` | Select Checkpoint |
| ControlNet Selector | `YogurtControlNetSelector` | `YogurtNodes/Models` | Select ControlNet |
| Convert LoRA Keys | `YogurtConvertLoraKeys` | `YogurtNodes/Models` | Rename LoRA keys by mapping JSON. |
| Create LoRA Mapping JSON | `YogurtCreateLoraMappingJson` | `YogurtNodes/Models` | Build a best-effort mapping from LoRA A keys to LoRA B keys. |
| Diffusion Model Selector | `YogurtDiffusionModelSelector` | `YogurtNodes/Models` | Select Diffusion Model |
| LoRA Add (Rank Aware) | `YogurtLoraAdd` | `YogurtNodes/Models` | Merge two LoRAs, with SVD rank alignment when ranks differ. |
| LoRA Compress | `YogurtLoraRankCompress` | `YogurtNodes/Models` | Compress LoRA rank with SVD for standard .lora_down/.lora_up pairs. Optionally absorb alpha/rank first to preserve the actual LoRA effect before compression. |
| LoRA Layers Operation | `YogurtLoraLayersOperation` | `YogurtNodes/Models` | Modify only selected LoRA layers by index. |
| LoRA Load Only | `YogurtLoadLoraOnly` | `YogurtNodes/Models` | Load a LoRA without applying it. Use with other LoRA operation nodes. |
| LoRA Merge Full Rank | `YogurtLoraMerge` | `YogurtNodes/Models` | Merge up to five standard LoRAs exactly by concatenating rank dimensions. Fast and preserves the summed model-side effect exactly, but output rank/file size grow. Does not support DoRA or LoCon/reshape variants. |
| LoRA Scale Alpha | `YogurtLoraScaleAlpha` | `YogurtNodes/Models` | Scale only LoRA alpha metadata so the adjusted LoRA can be saved downstream. |
| LoRA Scale Weights | `YogurtLoraScaleWeights` | `YogurtNodes/Models` | Scale LoRA tensor weights globally so effect can be tuned while using strength=1. |
| LoRA Simple Add | `YogurtLoraSimpleAdd` | `YogurtNodes/Models` | Simple weighted sum of two LoRA states. |
| LoRA Stat Viewer | `YogurtLoraStatViewer` | `YogurtNodes/Models` | Inspect LoRA key patterns to help define regex and layer selection. |
| Lora Selector | `YogurtLoraSelector` | `YogurtNodes/Models` | Select Lora |
| Merge LoRA To Model | `YogurtMergeLoraToModel` | `YogurtNodes/Models` | Apply loaded LoRA to model and optional CLIP. |
| Save LoRA | `YogurtSaveLora` | `YogurtNodes/Models` | Save LoRA state as safetensors. |


### 输入/输出操作节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Any Bridge | `YogurtAnyBridge` | `YogurtNodes/IO` | Any Bridge |
| Create Directory | `YogurtCreateDirectory` | `YogurtNodes/IO` | Create a directory |
| Create Parent Directory | `YogurtCreateParentDirectory` | `YogurtNodes/IO` | Create a parent directory |
| Deserialize Any | `YogurtDeserializeAny` | `YogurtNodes/IO` | Deserialize bytes data to Python object using pickle |
| Glob Files | `YogurtGlobFiles` | `YogurtNodes/IO` | Use glob pattern to traverse the folder, return the matching file path list |
| Load Audio Path | `YogurtLoadAudioPath` | `YogurtNodes/IO` | Load audio from path. |
| Load Bytes | `YogurtLoadBytes` | `YogurtNodes/IO` | Load bytes data from a file |
| Load Image | `YogurtLoadImage` | `YogurtNodes/IO` | Load image. |
| Load Image Path | `YogurtLoadImagePath` | `YogurtNodes/IO` | Load image from path. |
| Load Video | `YogurtLoadVideo` | `YogurtNodes/IO` | Load video. |
| Load Video Path | `YogurtLoadVideoPath` | `YogurtNodes/IO` | Load video from path. |
| Path Operator | `YogurtPathOperator` | `YogurtNodes/IO` | Execute join, relative, or common path operations. |
| Preview Any Bridge | `YogurtPreviewAnyBridge` | `YogurtNodes/IO` | Preview Any Bridge |
| Preview Any Bridge (Output) | `YogurtPreviewAnyBridgeOutput` | `YogurtNodes/IO` | Preview Any Bridge (Output) node. |
| Preview Image Bridge | `YogurtPreviewImageBridge` | `YogurtNodes/IO` | Preview the input images. |
| Preview Image Bridge (Output) | `YogurtPreviewImageBridgeOutput` | `YogurtNodes/IO` | Preview Image Bridge (Output) node. |
| Preview Mask Bridge | `YogurtPreviewMaskBridge` | `YogurtNodes/IO` | Preview the input masks. |
| Preview Mask Bridge (Output) | `YogurtPreviewMaskBridgeOutput` | `YogurtNodes/IO` | Preview Mask Bridge (Output) node. |
| Save Bytes Bridge | `YogurtSaveBytesBridge` | `YogurtNodes/IO` | Saves the input bytes data to your ComfyUI output directory. |
| Save Bytes Bridge (Non Output) | `YogurtSaveBytesBridgeNonOutput` | `YogurtNodes/IO` | Save Bytes Bridge (Non Output) node. |
| Save Image Bridge | `YogurtSaveImageBridge` | `YogurtNodes/IO` | Saves the input images to your ComfyUI output directory. |
| Save Image Bridge (Non Output) | `YogurtSaveImageBridgeNonOutput` | `YogurtNodes/IO` | Save Image Bridge (Non Output) node. |
| Save Image Bridge Ex | `YogurtSaveImageBridgeEx` | `YogurtNodes/IO` | Saves the input images to your ComfyUI output directory. |
| Save Image Bridge Ex (Non Output) | `YogurtSaveImageBridgeExNonOutput` | `YogurtNodes/IO` | Save Image Bridge Ex (Non Output) node. |
| Save Image Bridge Simple | `YogurtSaveImageBridgeSimple` | `YogurtNodes/IO` | Saves the input images to your ComfyUI output directory. |
| Save Image Bridge Simple (Non Output) | `YogurtSaveImageBridgeSimpleNonOutput` | `YogurtNodes/IO` | Save Image Bridge Simple (Non Output) node. |
| Save Mask Bridge | `YogurtSaveMaskBridge` | `YogurtNodes/IO` | Saves the input masks to your ComfyUI output directory. |
| Save Mask Bridge | `YogurtSaveMaskBridgeEx` | `YogurtNodes/IO` | Saves the input masks to your ComfyUI output directory. |
| Save Mask Bridge | `YogurtSaveMaskBridgeSimple` | `YogurtNodes/IO` | Saves the input masks to your ComfyUI output directory. |
| Save Mask Bridge (Non Output) | `YogurtSaveMaskBridgeExNonOutput` | `YogurtNodes/IO` | Save Mask Bridge (Non Output) node. |
| Save Mask Bridge (Non Output) | `YogurtSaveMaskBridgeNonOutput` | `YogurtNodes/IO` | Save Mask Bridge (Non Output) node. |
| Save Mask Bridge Simple (Non Output) | `YogurtSaveMaskBridgeSimpleNonOutput` | `YogurtNodes/IO` | Save Mask Bridge Simple (Non Output) node. |
| Save Text Bridge | `YogurtSaveTextBridge` | `YogurtNodes/IO` | Saves the input text to your ComfyUI output directory. |
| Save Text Bridge (Non Output) | `YogurtSaveTextBridgeNonOutput` | `YogurtNodes/IO` | Save Text Bridge (Non Output) node. |
| Serialize Any | `YogurtSerializeAny` | `YogurtNodes/IO` | Serialize any Python object to bytes using pickle |
| Split Path | `YogurtSplitPath` | `YogurtNodes/IO` | Split path to parts |


### 语言模型节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Chat (Custom Model) | `YogurtCustomModelChat` | `YogurtNodes/LLM` | Chat with any model id on the connected provider. |
| Claude Chat | `YogurtClaudeChat` | `YogurtNodes/LLM` | Anthropic Claude via OpenRouter or an OpenAI-compatible relay. |
| FreedomGPT Generate Image (Legacy) | `YogurtFreedomGPTGenerateImage` | `YogurtNodes/LLM` | Generate images using FreedomGPT API |
| FreedomGPT Generate Text (Legacy) | `YogurtFreedomGPTGenerateText` | `YogurtNodes/LLM` | Generate text using FreedomGPT API |
| FreedomGPT Image Understand (Legacy) | `YogurtFreedomGPTImageUnderstand` | `YogurtNodes/LLM` | Understand image content using FreedomGPT vision models |
| GPT Chat | `YogurtGPTChat` | `YogurtNodes/LLM` | OpenAI GPT text generation with optional image understanding and chat history. |
| GPT Image | `YogurtGPTImageGenerateImage` | `YogurtNodes/LLM` | OpenAI GPT Image generation and editing (2.5 / 2 / 1.5 / 1 / 1 Mini). |
| GRSAI Generate Image (Legacy) | `YogurtGRSAIGenerateImage` | `YogurtNodes/LLM` | Generate or edit images with the GRSAI API and return torch tensors |
| Gemini Chat | `YogurtGeminiChat` | `YogurtNodes/LLM` | Gemini text generation with optional image understanding and chat history. |
| Gemini Generate Image (Legacy) | `YogurtGeminiGenerateImage` | `YogurtNodes/LLM` | Generate image using Gemini API and return as torch.Tensor (h,w,c) and text |
| Gemini Generate Text (Legacy) | `YogurtGeminiGenerateText` | `YogurtNodes/LLM` | Generate text using Gemini API |
| Gemini Image Understand (Legacy) | `YogurtGeminiImageUnderstand` | `YogurtNodes/LLM` | Understand images using Gemini API |
| Grok Chat | `YogurtGrokChat` | `YogurtNodes/LLM` | xAI Grok text generation with optional image understanding and chat history. |
| Grok Generate Image (Legacy) | `YogurtGrokGenerateImage` | `YogurtNodes/LLM` | Generate image using xAI Grok API and return as torch.Tensor (h,w,c) and text |
| Grok Generate Text (Legacy) | `YogurtGrokGenerateText` | `YogurtNodes/LLM` | Generate text using xAI API |
| Grok Image Understand (Legacy) | `YogurtGrokImageUnderstand` | `YogurtNodes/LLM` | Understand image content using xAI vision models |
| Grok Imagine | `YogurtGrokImagineGenerateImage` | `YogurtNodes/LLM` | xAI Grok Imagine image generation and editing. |
| History Builder | `YogurtHistoryBuilder` | `YogurtNodes/LLM` | 构建与 LLM 节点兼容的会话历史 |
| Image Generation (Custom Model) | `YogurtCustomModelGenerateImage` | `YogurtNodes/LLM` | Image generation with any model id on the connected provider. |
| Magnific Creative Upscale | `YogurtMagnificCreativeUpscale` | `YogurtNodes/LLM` | Magnific creative upscaling with prompt, creativity and engine controls. |
| Magnific Image Upscale API (Legacy) | `YogurtMagnificImageUpscaleAPI` | `YogurtNodes/LLM` | Call the Magnific image upscaler API, wait for completion, and return an IMAGE batch. |
| Magnific Precision Upscale | `YogurtMagnificPrecisionUpscale` | `YogurtNodes/LLM` | Magnific precision upscaling (V2 with flavors, or V1). |
| Nano Banana | `YogurtNanoBananaGenerateImage` | `YogurtNodes/LLM` | Nano Banana image generation and editing (2.1 / 2 / 2 Lite / Pro / 1). |
| OpenAI Generate Image (Legacy) | `YogurtOpenAIGenerateImage` | `YogurtNodes/LLM` | Generate image using OpenAI API and return as torch.Tensor (h,w,c) and text |
| OpenAI Generate Text (Legacy) | `YogurtOpenAIGenerateText` | `YogurtNodes/LLM` | Generate text using OpenAI API |
| OpenAI Image Understand (Legacy) | `YogurtOpenAIImageUnderstand` | `YogurtNodes/LLM` | Understand image content using OpenAI vision models |
| OpenRouter Generate Image (Legacy) | `YogurtOpenRouterGenerateImage` | `YogurtNodes/LLM` | Generate image using OpenRouter API and return as torch.Tensor (h,w,c) and text |
| OpenRouter Generate Text (Legacy) | `YogurtOpenRouterGenerateText` | `YogurtNodes/LLM` | Generate text using OpenRouter API |
| OpenRouter Image Understand (Legacy) | `YogurtOpenRouterImageUnderstand` | `YogurtNodes/LLM` | Understand image content using OpenRouter API |
| Provider: Alibaba DashScope | `YogurtDashScopeProvider` | `YogurtNodes/LLM` | Alibaba Cloud DashScope (Bailian) provider for Qwen Image and Wan. |
| Provider: BytePlus ModelArk | `YogurtBytePlusArkProvider` | `YogurtNodes/LLM` | BytePlus ModelArk (international) provider for Seedream models. |
| Provider: FreedomGPT | `YogurtFreedomGPTProvider` | `YogurtNodes/LLM` | FreedomGPT provider; use with the custom-model chat and image nodes. |
| Provider: GRSAI | `YogurtGRSAIProvider` | `YogurtNodes/LLM` | GRSAI drawing relay provider (Nano Banana / GPT Image). |
| Provider: Google AI Studio | `YogurtGoogleAIStudioProvider` | `YogurtNodes/LLM` | Google AI Studio (Gemini API) provider; connect to Gemini / Nano Banana nodes. |
| Provider: Google GenAI Compatible | `YogurtGoogleGenAICompatibleProvider` | `YogurtNodes/LLM` | Any Gemini-API-compatible endpoint with a custom base URL. |
| Provider: Magnific | `YogurtMagnificProvider` | `YogurtNodes/LLM` | Magnific upscaler API provider. |
| Provider: OpenAI | `YogurtOpenAIProvider` | `YogurtNodes/LLM` | Official OpenAI API provider. |
| Provider: OpenAI Compatible | `YogurtOpenAICompatibleProvider` | `YogurtNodes/LLM` | Any OpenAI-compatible endpoint (chat completions / images) with a custom base URL. |
| Provider: OpenRouter | `YogurtOpenRouterProvider` | `YogurtNodes/LLM` | OpenRouter provider with optional upstream provider routing. |
| Provider: Topaz Labs | `YogurtTopazProvider` | `YogurtNodes/LLM` | Topaz Labs Image API provider. |
| Provider: Vertex AI | `YogurtVertexAIProvider` | `YogurtNodes/LLM` | Vertex AI provider (express-mode API key or service-account credentials). |
| Provider: Volcengine Ark | `YogurtVolcengineArkProvider` | `YogurtNodes/LLM` | Volcengine Ark (China) provider for Seedream and Doubao models. |
| Provider: xAI | `YogurtXAIProvider` | `YogurtNodes/LLM` | xAI (Grok) API provider. |
| Qwen Generate/Edit Image (Legacy) | `YogurtQwenGenerateImage` | `YogurtNodes/LLM` | 使用阿里云百炼 Qwen 图片模型进行文生图或多图编辑 |
| Qwen Image | `YogurtQwenImageGenerateImage` | `YogurtNodes/LLM` | Alibaba Qwen Image generation and editing. |
| SeeDream Generate Image (Legacy) | `YogurtSeeDreamGenerateImage` | `YogurtNodes/LLM` | 使用豆包SeeDream API生成图像，支持文生图、图生图、多图生图和序列图像生成 |
| Seedream | `YogurtSeedreamGenerateImage` | `YogurtNodes/LLM` | ByteDance Seedream image generation and editing (5.0 Pro / Flash / Lite, 4.5, 4.0). |
| Topaz Generative Upscale | `YogurtTopazGenerativeUpscale` | `YogurtNodes/LLM` | Topaz generative upscaling (Wonder 3.5, Bloom 2, Reimagine, Redefine, Recovery V2). |
| Topaz Image Upscale API (Legacy) | `YogurtTopazImageUpscaleAPI` | `YogurtNodes/LLM` | Call the Topaz Labs Image API, wait for completion, and return an IMAGE batch. |
| Topaz Upscale | `YogurtTopazUpscale` | `YogurtNodes/LLM` | Topaz precision upscaling (Standard V2, Low Resolution V2, High Fidelity V2, CGI, Text Refine). |
| Vertex AI Generate Image (Legacy) | `YogurtVertexAIGenerateImage` | `YogurtNodes/LLM` | Generate image using Vertex AI API and return as torch.Tensor (h,w,c) and text |
| Vertex AI Generate Text (Legacy) | `YogurtVertexAIGenerateText` | `YogurtNodes/LLM` | Generate text using Vertex AI |
| Vertex Image Understand (Legacy) | `YogurtVertexAIImageUnderstand` | `YogurtNodes/LLM` | Understand images using Vertex AI |
| Wan Generate/Edit Image (Legacy) | `YogurtWanGenerateImage` | `YogurtNodes/LLM` | 使用阿里云百炼 Wan 图片模型进行文生图或图像编辑 |
| Wan Image | `YogurtWanImageGenerateImage` | `YogurtNodes/LLM` | Alibaba Wan image generation and editing. |


### 网络节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| ComfyUI Client Get Output | `YogurtComfyUIClientGetOutput` | `YogurtNodes/Net` | 根据节点 ID/名称，从结果包中取出该节点的全部输出列表 |
| ComfyUI Client Load | `YogurtComfyUIClientLoad` | `YogurtNodes/Net` | 配置 ComfyUI 客户端实例，供后续节点复用 |
| ComfyUI Client Run | `YogurtComfyUIClientRun` | `YogurtNodes/Net` | 提交工作流并等待结果返回 |
| ComfyUI Client Set Float | `YogurtComfyUIClientSetFloat` | `YogurtNodes/Net` | 向工作流节点输入设置浮点数 |
| ComfyUI Client Set Image | `YogurtComfyUIClientSetImage` | `YogurtNodes/Net` | 上传图片并写入工作流节点输入 |
| ComfyUI Client Set Int | `YogurtComfyUIClientSetInt` | `YogurtNodes/Net` | 向工作流节点输入设置整数 |
| ComfyUI Client Set Seed | `YogurtComfyUIClientSetSeed` | `YogurtNodes/Net` | 为工作流中的节点设置随机种子 |
| ComfyUI Client Set String | `YogurtComfyUIClientSetString` | `YogurtNodes/Net` | 向工作流节点输入设置字符串 |

## 🔑 Gemini API Key 配置说明

使用 Gemini 相关节点前，您需要获取并配置 Gemini API Key。支持以下三种方式，优先级如下：

1. **代码参数传递**
   - 直接在代码中初始化 GeminiClient 时传入 `api_key` 参数（优先级最高）。

2. **api_key.json 文件**
   - 在 `custom_nodes/ComfyUI-YogurtNodes/yogurt_nodes/llm/` 目录下创建 `api_key.json` 文件，内容如下：
     ```json
     {
       "gemini": "你的API密钥"
     }
     ```
   - 仅当未通过代码参数传递时才会读取。

3. **环境变量**
   - 设置环境变量 `GEMINI_API_KEY`，仅当前两者都未设置时才会读取。
   - 示例（Windows 命令行）：
     ```cmd
     set GEMINI_API_KEY=你的API密钥
     ```

如未正确配置 API Key，相关节点将无法正常使用。API Key 可在 [Google AI Studio](https://aistudio.google.com/app/apikey) 获取。

## 🔑 OpenAI API Key 配置说明

使用 OpenAI 相关节点前，您需要获取并配置 OpenAI API Key。支持以下三种方式，优先级如下：

1. **代码参数传递**
   - 直接在代码中初始化 OpenAIClient 时传入 `api_key` 参数（优先级最高）。

2. **api_key.json 文件**
   - 在 `custom_nodes/ComfyUI-YogurtNodes/yogurt_nodes/llm/` 目录下创建 `api_key.json` 文件，内容如下：
     ```json
     {
       "openai": "你的API密钥",
       "openai_base_url": "https://api.openai.com/v1"
     }
     ```
   - `openai_base_url` 是可选的，默认为官方OpenAI API。
   - 仅当未通过代码参数传递时才会读取。

3. **环境变量**
   - 设置环境变量 `OPENAI_API_KEY` 和可选的 `OPENAI_BASE_URL`，仅当前两者都未设置时才会读取。
   - 示例（Windows 命令行）：
     ```cmd
     set OPENAI_API_KEY=你的API密钥
     set OPENAI_BASE_URL=https://api.openai.com/v1
     ```

### 自定义基础URL支持

OpenAI节点支持自定义基础URL，使其兼容：
- 官方OpenAI API
- Azure OpenAI服务
- OpenAI兼容API（如LocalAI、Ollama等）
- 自托管OpenAI兼容服务器

只需将 `base_url` 参数设置为您首选的端点。

如未正确配置 API Key，OpenAI节点将无法正常使用。API Key 可在 [OpenAI Platform](https://platform.openai.com/api-keys) 获取。

## 🔑 OpenRouter API Key 配置说明

使用 OpenRouter 相关节点前，您需要获取并配置 OpenRouter API Key。支持以下三种方式，优先级如下：

1. **代码参数传递**
   - 直接在代码中初始化 OpenRouterClient 时传入 `api_key` 参数（优先级最高）。

2. **api_key.json 文件**
   - 在 `custom_nodes/ComfyUI-YogurtNodes/yogurt_nodes/llm/` 目录下创建 `api_key.json` 文件，内容如下：
     ```json
     {
       "openrouter": "你的API密钥"
     }
     ```
   - 仅当未通过代码参数传递时才会读取。

3. **环境变量**
   - 设置环境变量 `OPENROUTER_API_KEY`，仅当前两者都未设置时才会读取。
   - 示例（Windows 命令行）：
     ```cmd
     set OPENROUTER_API_KEY=你的API密钥
     ```

如未正确配置 API Key，OpenRouter节点将无法正常使用。API Key 可在 [OpenRouter Platform](https://openrouter.ai/keys) 获取。

## 🧩 LLM / 生图 / 放大节点：供应商与模型

LLM 相关节点分为两类，连线使用：

- **模型节点**（`YogurtNodes/LLM/Image`、`Text`、`Upscale`）：每个模型系列一个节点，例如 Nano Banana、GPT Image、Seedream、Gemini Chat、Claude Chat、Topaz Upscale。`model` 下拉切换版本，下方参数随版本变化，只显示该版本真正支持的参数。
- **供应商节点**（`YogurtNodes/LLM/Providers`）：只负责接入配置（key、base_url、代理、超时），输出 `provider` 连到模型节点。同一个模型可以换不同供应商调用，例如 Nano Banana 可走 Google AI Studio、Vertex AI、OpenRouter 或 GRSAI。

使用要点：

- 模型节点的 `provider` 不连线时，使用该系列的默认官方渠道，key 从下表的 `api_key.json` 键名或环境变量读取。
- 连接了不支持该模型的供应商时，节点会报错并列出可用的供应商。
- **OpenAI Compatible** 与 **Google GenAI Compatible** 用于自填 base_url 的中转，必须在节点里填写 `api_key`，不会回退到 `api_key.json` 里的官方 key。
- 参考图输入是动态数量的：连上一张后自动出现下一个输入口。
- `timeout` 为 0 表示不限时；生图通常需要几十秒到几分钟。
- 需要传尚未暴露的新参数时，在高级选项的 `extra` 里填 JSON 对象。
- 新模型尚未内置时，可以用各系列下拉里的 `Custom model`，或 `Chat (Custom Model)` / `Image Generation (Custom Model)` 节点直接填模型 ID。
- 旧版节点保留在 `YogurtNodes/LLM/Legacy`，名称带 `(Legacy)`，已有工作流可继续使用。

| 供应商节点 | `api_key.json` 键名 | 环境变量 | 不连线时作为默认的模型节点 |
| --- | --- | --- | --- |
| Google AI Studio | `gemini` | `GEMINI_API_KEY` | Nano Banana、Gemini Chat |
| Vertex AI | `vertex_ai_json`、`vertex_ai_project`、`vertex_ai_region` | `GOOGLE_APPLICATION_CREDENTIALS` | — |
| Google GenAI Compatible | 必须在节点中填写 | — | — |
| OpenAI | `openai`、`openai_base_url` | `OPENAI_API_KEY`、`OPENAI_BASE_URL` | GPT Image、GPT Chat、自定义模型节点 |
| OpenAI Compatible | 必须在节点中填写 | — | — |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | Claude Chat |
| xAI | `xai`（或 `grok`）、`xai_base_url` | `XAI_API_KEY`（或 `GROK_API_KEY`） | Grok Imagine、Grok Chat |
| GRSAI | `grsai`、`grsai_base_url` | `GRSAI_API_KEY` | — |
| FreedomGPT | `freedomgpt` | `FREEDOMGPT_API_KEY` | — |
| Volcengine Ark（火山方舟） | `seedream`（或 `ark`） | `ARK_API_KEY` | Seedream |
| BytePlus ModelArk | `byteplus` | `BYTEPLUS_API_KEY` | — |
| Alibaba DashScope（百炼） | `dashscope`（或 `qwen`、`wan`）、`dashscope_base_url` | `DASHSCOPE_API_KEY` | Qwen Image、Wan Image |
| Topaz Labs | `topaz`、`topaz_base_url` | `TOPAZ_API_KEY` | Topaz Upscale、Topaz Generative Upscale |
| Magnific | `magnific`、`magnific_base_url` | `MAGNIFIC_API_KEY` | Magnific Creative / Precision Upscale |

新增供应商或模型的方法见 [yogurt_nodes/llm/EXTENDING.md](yogurt_nodes/llm/EXTENDING.md)。

## 🤝 贡献

欢迎提交PR来帮助改进项目！

## 📄 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件。

## 📞 联系方式

如有问题、bug反馈或功能建议，请[提交Issue](https://github.com/yogurt7771/ComfyUI-YogurtNodes/issues)。

## 🙏 致谢

- ComfyUI社区
- 所有贡献者
