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

当前导出节点总数：**146**。

| 分组 | 数量 |
| --- | ---: |
| 图像处理节点 | 12 |
| 数字处理节点 | 2 |
| 字符串处理节点 | 9 |
| 逻辑处理节点 | 41 |
| 模型节点 | 16 |
| 输入/输出操作节点 | 35 |
| 语言模型节点 | 23 |
| 网络节点 | 8 |

### 图像处理节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Add Text To Image | `YogurtAddTextToImage` | `YogurtNodes/Image` | Add text to image. |
| Batch Images | `YogurtBatchImages` | `YogurtNodes/Image` | Batch images. |
| Get Image Size | `YogurtGetImageSize` | `YogurtNodes/Image` | Get image size information. |
| Image Crop By Mask | `YogurtImageCropByMask` | `YogurtNodes/Image` | Crop image to the minimum bounding box of the mask above threshold. |
| Image Scale To Total Pixels Advanced | `YogurtImageScaleToTotalPixelsAdvanced` | `YogurtNodes/Image` | Image Scale To Total Pixels Advanced. |
| Image Tile (Seam Mask) | `YogurtImageTileWithSeamMask` | `YogurtNodes/Image` | Split image into overlapped tiles and generate inpaint masks (white=inpaint, black=reference). |
| Image Untile (Seam Mask) | `YogurtImageUntileWithSeamMask` | `YogurtNodes/Image` | Merge overlapped tiles back to one image with seam feathering (mask + overlap-based smooth transition). |
| Magnific Image Upscale API | `YogurtMagnificImageUpscaleAPI` | `YogurtNodes/Image` | Call the Magnific image upscaler API, wait for completion, and return an IMAGE batch. |
| Poisson Blend | `YogurtPoissonBlend` | `YogurtNodes/Image` | 使用OpenCV泊松融合(seamlessClone)将前景融合到背景。 |
| Replace Image In Batch | `YogurtReplaceImageInBatch` | `YogurtNodes/Image` | Replace one image inside an image batch at the given index. |
| Tile Info To TTP Image Assy Args | `YogurtTileInfoToTTPImageAssyArgs` | `YogurtNodes/Image` | Convert tile_info to TTP_Image_Assy inputs: positions/original_size/grid_size/padding. |
| Topaz Image Upscale API | `YogurtTopazImageUpscaleAPI` | `YogurtNodes/Image` | Call the Topaz Labs Image API, wait for completion, and return an IMAGE batch. |


### 数字处理节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Range | `YogurtRange` | `YogurtNodes/Number` | get a number from a range |
| RangeItem | `YogurtRangeItem` | `YogurtNodes/Number` | get a value from a range |


### 字符串处理节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| Replace Delimiter | `YogurtReplaceDelimiter` | `YogurtNodes/String` | Replace delimiter in string. Support regex |
| Split Path | `YogurtSplitPath` | `YogurtNodes/String` | Split path to parts |
| String Concat | `YogurtStringConcat` | `YogurtNodes/String` | 拼接多个字符串，支持自定义分隔符和可变数量的输入 |
| String Format | `YogurtStringFormat` | `YogurtNodes/String` | Format strings |
| String Join | `YogurtStringJoin` | `YogurtNodes/String` | 将多个字符串使用指定连接符连接 |
| String Lines Count | `YogurtStringLinesCount` | `YogurtNodes/String` | Get the number of lines in a multiline string |
| String Lines Switch | `YogurtStringLinesSwitch` | `YogurtNodes/String` | Get line from multiline string by index |
| String To Value | `YogurtStringToValue` | `YogurtNodes/String` | Get value from string |
| Regex Node | `YogurtRegexNode` | `ZnzmoNodes/String` | Regex-based extraction and replacement for multiline text. |


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
| Diffusion Model Selector | `YogurtDiffusionModelSelector` | `YogurtNodes/Models` | Select Diffusion Model |
| Lora Selector | `YogurtLoraSelector` | `YogurtNodes/Models` | Select Lora |
| Convert LoRA Keys | `YogurtConvertLoraKeys` | `YogurtNodes/Models/LoRA` | Rename LoRA keys by mapping JSON. |
| Create LoRA Mapping JSON | `YogurtCreateLoraMappingJson` | `YogurtNodes/Models/LoRA` | Build a best-effort mapping from LoRA A keys to LoRA B keys. |
| LoRA Add (Rank Aware) | `YogurtLoraAdd` | `YogurtNodes/Models/LoRA` | Merge two LoRAs, with SVD rank alignment when ranks differ. |
| LoRA Compress | `YogurtLoraRankCompress` | `YogurtNodes/Models/LoRA` | Compress LoRA rank with SVD for standard .lora_down/.lora_up pairs. Optionally absorb alpha/rank first to preserve the actual LoRA effect before compression. |
| LoRA Layers Operation | `YogurtLoraLayersOperation` | `YogurtNodes/Models/LoRA` | Modify only selected LoRA layers by index. |
| LoRA Load Only | `YogurtLoadLoraOnly` | `YogurtNodes/Models/LoRA` | Load a LoRA without applying it. Use with other LoRA operation nodes. |
| LoRA Merge Full Rank | `YogurtLoraMerge` | `YogurtNodes/Models/LoRA` | Merge up to five standard LoRAs exactly by concatenating rank dimensions. Fast and preserves the summed model-side effect exactly, but output rank/file size grow. Does not support DoRA or LoCon/reshape variants. |
| LoRA Scale Weights | `YogurtLoraScaleWeights` | `YogurtNodes/Models/LoRA` | Scale LoRA tensor weights globally so effect can be tuned while using strength=1. |
| LoRA Simple Add | `YogurtLoraSimpleAdd` | `YogurtNodes/Models/LoRA` | Simple weighted sum of two LoRA states. |
| LoRA Stat Viewer | `YogurtLoraStatViewer` | `YogurtNodes/Models/LoRA` | Inspect LoRA key patterns to help define regex and layer selection. |
| Merge LoRA To Model | `YogurtMergeLoraToModel` | `YogurtNodes/Models/LoRA` | Apply loaded LoRA to model and optional CLIP. |
| Save LoRA | `YogurtSaveLora` | `YogurtNodes/Models/LoRA` | Save LoRA state as safetensors. |


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


### 语言模型节点

| 节点 | Class ID | 分类 | 说明 |
| --- | --- | --- | --- |
| FreedomGPT Generate Image | `YogurtFreedomGPTGenerateImage` | `YogurtNodes/LLM` | Generate images using FreedomGPT API |
| FreedomGPT Generate Text | `YogurtFreedomGPTGenerateText` | `YogurtNodes/LLM` | Generate text using FreedomGPT API |
| FreedomGPT Image Understand | `YogurtFreedomGPTImageUnderstand` | `YogurtNodes/LLM` | Understand image content using FreedomGPT vision models |
| GRSAI Generate Image | `YogurtGRSAIGenerateImage` | `YogurtNodes/LLM` | Generate or edit images with the GRSAI API and return torch tensors |
| Gemini Generate Image | `YogurtGeminiGenerateImage` | `YogurtNodes/LLM` | Generate image using Gemini API and return as torch.Tensor (h,w,c) and text |
| Gemini Generate Text | `YogurtGeminiGenerateText` | `YogurtNodes/LLM` | Generate text using Gemini API |
| Gemini Image Understand | `YogurtGeminiImageUnderstand` | `YogurtNodes/LLM` | Understand images using Gemini API |
| Grok Generate Image | `YogurtGrokGenerateImage` | `YogurtNodes/LLM` | Generate image using xAI Grok API and return as torch.Tensor (h,w,c) and text |
| Grok Generate Text | `YogurtGrokGenerateText` | `YogurtNodes/LLM` | Generate text using xAI API |
| Grok Image Understand | `YogurtGrokImageUnderstand` | `YogurtNodes/LLM` | Understand image content using xAI vision models |
| History Builder | `YogurtHistoryBuilder` | `YogurtNodes/LLM` | 构建与 LLM 节点兼容的会话历史 |
| OpenAI Generate Image | `YogurtOpenAIGenerateImage` | `YogurtNodes/LLM` | Generate image using OpenAI API and return as torch.Tensor (h,w,c) and text |
| OpenAI Generate Text | `YogurtOpenAIGenerateText` | `YogurtNodes/LLM` | Generate text using OpenAI API |
| OpenAI Image Understand | `YogurtOpenAIImageUnderstand` | `YogurtNodes/LLM` | Understand image content using OpenAI vision models |
| OpenRouter Generate Image | `YogurtOpenRouterGenerateImage` | `YogurtNodes/LLM` | Generate image using OpenRouter API and return as torch.Tensor (h,w,c) and text |
| OpenRouter Generate Text | `YogurtOpenRouterGenerateText` | `YogurtNodes/LLM` | Generate text using OpenRouter API |
| OpenRouter Image Understand | `YogurtOpenRouterImageUnderstand` | `YogurtNodes/LLM` | Understand image content using OpenRouter API |
| Qwen Generate/Edit Image | `YogurtQwenGenerateImage` | `YogurtNodes/LLM` | 使用阿里云百炼 Qwen 图片模型进行文生图或多图编辑 |
| SeeDream Generate Image | `YogurtSeeDreamGenerateImage` | `YogurtNodes/LLM` | 使用豆包SeeDream API生成图像，支持文生图、图生图、多图生图和序列图像生成 |
| Vertex AI Generate Image | `YogurtVertexAIGenerateImage` | `YogurtNodes/LLM` | Generate image using Vertex AI API and return as torch.Tensor (h,w,c) and text |
| Vertex AI Generate Text | `YogurtVertexAIGenerateText` | `YogurtNodes/LLM` | Generate text using Vertex AI |
| Vertex Image Understand | `YogurtVertexAIImageUnderstand` | `YogurtNodes/LLM` | Understand images using Vertex AI |
| Wan Generate/Edit Image | `YogurtWanGenerateImage` | `YogurtNodes/LLM` | 使用阿里云百炼 Wan 图片模型进行文生图或图像编辑 |


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

## 🤝 贡献

欢迎提交PR来帮助改进项目！

## 📄 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件。

## 📞 联系方式

如有问题、bug反馈或功能建议，请[提交Issue](https://github.com/yogurt7771/ComfyUI-YogurtNodes/issues)。

## 🙏 致谢

- ComfyUI社区
- 所有贡献者
