# ComfyUI-YogurtNodes

ComfyUI-YogurtNodes is a collection of custom nodes for ComfyUI, providing a series of practical image processing and workflow enhancement functionalities.

## ✨ Features

- Custom node integration
- Easy-to-use image processing functions
- Full compatibility with ComfyUI workflows
- Text and image manipulation capabilities
- Advanced string processing utilities
- Model management and selection tools
- Comprehensive I/O operations support
- Integrated Gemini API for language and image understanding
- Logic control nodes for complex workflows

## 📦 Installation

### Requirements

- ComfyUI (installed and running)
- Python 3.x
- Required Python packages:
  - numpy
  - pillow
  - google-genai (for Gemini nodes)
  - openai (for OpenAI and OpenRouter nodes)
  - requests (for API calls)
  - opencv-python (for Poisson Blend)

### Installation Steps

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/yogurt7771/ComfyUI-YogurtNodes.git
```

3. Install dependencies:
```bash
cd ComfyUI-YogurtNodes
pip install -r requirements.txt
```

## 🚀 Usage

1. Start ComfyUI
2. Look for "Yogurt Nodes" category in the node browser
3. Drag and drop desired nodes into your workflow

## 🔧 Available Nodes

All exported nodes are listed here. ComfyUI display names keep the " (Yogurt Nodes)" suffix at runtime.

This section is auto-generated from exported node classes and their docstrings. Run `python tools/generate_readme.py` to refresh.

Total exported nodes: **147**.

| Group | Count |
| --- | ---: |
| Image Processing Nodes | 12 |
| Number Processing Nodes | 2 |
| String Processing Nodes | 9 |
| Logic Processing Nodes | 41 |
| Model Nodes | 17 |
| I/O Operation Nodes | 35 |
| Language Model Nodes | 23 |
| Network Nodes | 8 |

### Image Processing Nodes

| Node | Class ID | Category | Description |
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


### Number Processing Nodes

| Node | Class ID | Category | Description |
| --- | --- | --- | --- |
| Range | `YogurtRange` | `YogurtNodes/Number` | get a number from a range |
| RangeItem | `YogurtRangeItem` | `YogurtNodes/Number` | get a value from a range |


### String Processing Nodes

| Node | Class ID | Category | Description |
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


### Logic Processing Nodes

| Node | Class ID | Category | Description |
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


### Model Nodes

| Node | Class ID | Category | Description |
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
| LoRA Scale Alpha | `YogurtLoraScaleAlpha` | `YogurtNodes/Models/LoRA` | Scale only LoRA alpha metadata so the adjusted LoRA can be saved downstream. |
| LoRA Scale Weights | `YogurtLoraScaleWeights` | `YogurtNodes/Models/LoRA` | Scale LoRA tensor weights globally so effect can be tuned while using strength=1. |
| LoRA Simple Add | `YogurtLoraSimpleAdd` | `YogurtNodes/Models/LoRA` | Simple weighted sum of two LoRA states. |
| LoRA Stat Viewer | `YogurtLoraStatViewer` | `YogurtNodes/Models/LoRA` | Inspect LoRA key patterns to help define regex and layer selection. |
| Merge LoRA To Model | `YogurtMergeLoraToModel` | `YogurtNodes/Models/LoRA` | Apply loaded LoRA to model and optional CLIP. |
| Save LoRA | `YogurtSaveLora` | `YogurtNodes/Models/LoRA` | Save LoRA state as safetensors. |


### I/O Operation Nodes

| Node | Class ID | Category | Description |
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


### Language Model Nodes

| Node | Class ID | Category | Description |
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


### Network Nodes

| Node | Class ID | Category | Description |
| --- | --- | --- | --- |
| ComfyUI Client Get Output | `YogurtComfyUIClientGetOutput` | `YogurtNodes/Net` | 根据节点 ID/名称，从结果包中取出该节点的全部输出列表 |
| ComfyUI Client Load | `YogurtComfyUIClientLoad` | `YogurtNodes/Net` | 配置 ComfyUI 客户端实例，供后续节点复用 |
| ComfyUI Client Run | `YogurtComfyUIClientRun` | `YogurtNodes/Net` | 提交工作流并等待结果返回 |
| ComfyUI Client Set Float | `YogurtComfyUIClientSetFloat` | `YogurtNodes/Net` | 向工作流节点输入设置浮点数 |
| ComfyUI Client Set Image | `YogurtComfyUIClientSetImage` | `YogurtNodes/Net` | 上传图片并写入工作流节点输入 |
| ComfyUI Client Set Int | `YogurtComfyUIClientSetInt` | `YogurtNodes/Net` | 向工作流节点输入设置整数 |
| ComfyUI Client Set Seed | `YogurtComfyUIClientSetSeed` | `YogurtNodes/Net` | 为工作流中的节点设置随机种子 |
| ComfyUI Client Set String | `YogurtComfyUIClientSetString` | `YogurtNodes/Net` | 向工作流节点输入设置字符串 |

## 🔑 Gemini API Key Setup

Before using Gemini-related nodes, you must obtain and configure your Gemini API Key. There are three supported methods, in the following order of priority:

1. **Code Argument**
   - Pass the `api_key` argument directly when initializing `GeminiClient` (highest priority).

2. **api_key.json File**
   - Create an `api_key.json` file in `custom_nodes/ComfyUI-YogurtNodes/yogurt_nodes/llm/` with the following content:
     ```json
     {
       "gemini": "YOUR_API_KEY"
     }
     ```
   - This will be used only if the code argument is not provided.

3. **Environment Variable**
   - Set the environment variable `GEMINI_API_KEY` (used only if the above two are not set).
   - Example (Windows command line):
     ```cmd
     set GEMINI_API_KEY=YOUR_API_KEY
     ```

If the API Key is not configured correctly, Gemini nodes will not work. You can obtain your API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 🔑 OpenAI API Key Setup

Before using OpenAI-related nodes, you must obtain and configure your OpenAI API Key. There are three supported methods, in the following order of priority:

1. **Code Argument**
   - Pass the `api_key` argument directly when initializing `OpenAIClient` (highest priority).

2. **api_key.json File**
   - Create an `api_key.json` file in `custom_nodes/ComfyUI-YogurtNodes/yogurt_nodes/llm/` with the following content:
     ```json
     {
       "openai": "YOUR_API_KEY",
       "openai_base_url": "https://api.openai.com/v1"
     }
     ```
   - The `openai_base_url` is optional and defaults to the official OpenAI API.
   - This will be used only if the code argument is not provided.

3. **Environment Variable**
   - Set the environment variables `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` (used only if the above two are not set).
   - Example (Windows command line):
     ```cmd
     set OPENAI_API_KEY=YOUR_API_KEY
     set OPENAI_BASE_URL=https://api.openai.com/v1
     ```

### Custom Base URL Support

The OpenAI nodes support custom base URLs, making them compatible with:
- Official OpenAI API
- Azure OpenAI Service
- OpenAI-compatible APIs (like LocalAI, Ollama, etc.)
- Self-hosted OpenAI-compatible servers

Simply set the `base_url` parameter to your preferred endpoint.

If the API Key is not configured correctly, OpenAI nodes will not work. You can obtain your API Key from [OpenAI Platform](https://platform.openai.com/api-keys).

## 🔑 OpenRouter API Key Setup

Before using OpenRouter-related nodes, you must obtain and configure your OpenRouter API Key. There are three supported methods, in the following order of priority:

1. **Code Argument**
   - Pass the `api_key` argument directly when initializing `OpenRouterClient` (highest priority).

2. **api_key.json File**
   - Create an `api_key.json` file in `custom_nodes/ComfyUI-YogurtNodes/yogurt_nodes/llm/` with the following content:
     ```json
     {
       "openrouter": "YOUR_API_KEY"
     }
     ```
   - This will be used only if the code argument is not provided.

3. **Environment Variable**
   - Set the environment variable `OPENROUTER_API_KEY` (used only if the above two are not set).
   - Example (Windows command line):
     ```cmd
     set OPENROUTER_API_KEY=YOUR_API_KEY
     ```

If the API Key is not configured correctly, OpenRouter nodes will not work. You can obtain your API Key from [OpenRouter Platform](https://openrouter.ai/keys).

## 🤝 Contributing

Pull Requests are welcome to help improve the project!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, bug reports, or feature requests, please [open an issue](https://github.com/yogurt7771/ComfyUI-YogurtNodes/issues).

## 🙏 Acknowledgments

- ComfyUI Community
- All Contributors 
