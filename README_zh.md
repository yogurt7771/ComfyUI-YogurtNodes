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
- 逻辑控制节点支持复杂工作流

## 📦 安装

### 要求

- ComfyUI（已安装并运行）
- Python 3.x
- 需要的Python包：
  - numpy
  - pillow
  - google-generativeai (对于Gemini节点)

### 安装步骤

1. 导航到您的ComfyUI自定义节点目录：

```bash
cd custom_nodes
```

2. 克隆此仓库：

```bash
git clone https://github.com/[your-username]/ComfyUI-YogurtNodes.git
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

所有节点都标有"YogurtNodes"前缀，便于在ComfyUI界面中识别。

### 图像处理节点

#### Batch Images

- **类别：** YogurtNodes/Image
- **描述：** 将多个图像合并成一个批次，具有可自定义设置
- **特点：**
  - 支持多达16个输入图像
  - 多种插值方法（最近邻、双线性、双三次等）
  - 各种调整大小方法（拉伸、填充/裁剪、填充）
  - 可自定义填充值
  - 灵活的批处理切片，具有起始、结束和步长参数

#### Add Text To Image

- **类别：** YogurtNodes/Image
- **描述：** 为图像添加文本覆盖，具有广泛的自定义选项
- **特点：**
  - 文本可放置在顶部或底部
  - 中心对齐选项
  - 多种字体支持
  - 可自定义字体大小
  - 文本和背景颜色配置
  - 支持多行文本，自动换行

#### None Image

- **类别：** YogurtNodes/Image
- **描述：** 一个返回None作为图像输出的实用节点
- **用例：** 适用于需要空图像输出的条件工作流

### 数字处理节点

#### Range
- **类别：** YogurtNodes/Number
- **描述：** 生成指定区间的数字序列
- **特点：**
  - 支持自定义起始、结束、步长或数量
  - 输出torch张量、步数和步长

#### RangeItem
- **类别：** YogurtNodes/Number
- **描述：** 从区间序列中提取指定索引的数值
- **特点：**
  - 支持负索引
  - 输出字符串、整数和浮点数

### 字符串处理节点

#### String Lines Count

- **类别：** YogurtNodes/String
- **描述：** 计算多行字符串中的行数
- **特点：**
  - 可选行去除空格
  - 空行过滤控制
  - 可配置的计数行为

#### String To Value

- **类别：** YogurtNodes/String
- **描述：** 将字符串输入转换为多种值类型
- **特点：**
  - 自动转换为整数
  - 自动转换为浮点数
  - 保留原始字符串
  - 安全转换，具有回退值

#### String Lines Switch

- **类别：** YogurtNodes/String
- **描述：** 通过索引从多行字符串中提取特定行
- **特点：**
  - 行索引
  - 多种输出格式（字符串、整数、浮点数）
  - 行数输出
  - 空行处理选项

#### Replace Delimiter

- **类别：** YogurtNodes/String
- **描述：** 用自定义分隔符替换字符串中的分隔符
- **特点：**
  - 支持正则表达式
  - 自定义分隔符指定
  - 灵活的替换选项

#### Split Path

- **类别：** YogurtNodes/String
- **描述：** 将文件路径分割成其组成部分
- **特点：**
  - 路径组件提取
  - 返回父目录
  - 文件名和扩展名分离
  - 扩展名的多种格式选项

#### Regex Node
- **类别：** YogurtNodes/String
- **描述：** 基于正则表达式的多行文本提取与替换
- **特点：**
  - 支持extract和replace两种模式
  - 支持分组引用与格式化
  - 可按数量、顺序提取或替换
  - 适用于复杂文本处理

### 逻辑处理节点

#### None Node

- **类别：** YogurtNodes/Logic
- **描述：** 返回None值的实用节点
- **用例：** 适用于需要空值的条件工作流

#### Pack Any

- **类别：** YogurtNodes/Logic
- **描述：** 将多个输入项打包成单一输出
- **特点：**
  - 支持多达8个输入项
  - 处理任何类型的数据
  - 简化复杂工作流

#### Unpack Any

- **类别：** YogurtNodes/Logic
- **描述：** 将打包的输入项解包成多个输出
- **特点：**
  - 与Pack Any节点无缝配合
  - 输出多达8个独立项
  - 适用于数据分发

#### Switch

- **类别：** YogurtNodes/Logic
- **描述：** 基于条件的开关节点
- **特点：**
  - 支持正则表达式匹配
  - 多达8个条件分支
  - 默认值选项
  - 灵活的条件控制

### 模型选择节点

#### ControlNet Selector

- **类别：** YogurtNodes/Models
- **描述：** 高级ControlNet模型选择和配置
- **特点：**
  - 从可用ControlNets动态生成模型列表
  - 可调整强度参数
  - 起始和结束百分比控制
  - 模型路径和名称提取
  - 条件工作流的None选项

#### Diffusion Model Selector

- **类别：** YogurtNodes/Models
- **描述：** Stable Diffusion模型选择工具
- **特点：**
  - 可用扩散模型的动态列表
  - 触发词支持
  - 模型路径和名称提取
  - 条件工作流的None选项

#### Checkpoint Selector

- **类别：** YogurtNodes/Models
- **描述：** 检查点模型选择和管理
- **特点：**
  - 动态检查点列表
  - 触发词支持
  - 模型信息提取
  - 条件工作流的None选项

#### Lora Selector

- **类别：** YogurtNodes/Models
- **描述：** LoRA模型选择和配置
- **特点：**
  - 动态LoRA模型列表
  - 单独的模型和CLIP强度控制
  - 触发词支持
  - 模型路径和名称提取
  - 条件工作流的None选项

### 输入/输出操作节点

#### Save Image Bridge Ex

- **类别：** YogurtNodes/IO
- **描述：** 增强的图像保存功能，具有广泛的选项
- **特点：**
  - 多种格式支持（PNG、JPEG）
  - 可自定义输出目录
  - 带变量的动态文件名前缀
  - 元数据控制
  - 覆盖保护
  - 压缩级别控制
  - JPEG的质量设置
  - 自定义后缀支持

#### Save Image Bridge

- **类别：** YogurtNodes/IO
- **描述：** 基本图像保存功能
- **特点：**
  - PNG和JPEG支持
  - 基本输出选项
  - 元数据处理
  - 压缩设置

#### Preview Image Bridge

- **类别：** YogurtNodes/IO
- **描述：** 快速图像预览功能
- **特点：**
  - 临时文件处理
  - 快速预览生成
  - 自动清理

#### Save Text Bridge

- **类别：** YogurtNodes/IO
- **描述：** 文本文件保存，支持多种格式
- **特点：**
  - 多种格式支持（.txt、.json、.md、.csv）
  - 自定义文件扩展名
  - 动态文件名生成
  - 覆盖保护
  - UTF-8编码支持

#### Any Bridge

- **类别：** YogurtNodes/IO
- **描述：** 通用数据桥，用于工作流控制
- **特点：**
  - 处理任何数据类型
  - 可选的黑洞模式
  - 工作流控制支持

#### Create Directory

- **类别：** YogurtNodes/IO
- **描述：** 目录创建工具
- **特点：**
  - 创建单个目录
  - 自动父目录创建
  - 路径验证

#### Create Parent Directory

- **类别：** YogurtNodes/IO
- **描述：** 父目录创建工具
- **特点：**
  - 创建父目录
  - 递归创建支持
  - 路径验证
  - 已存在目录的安全操作

#### Save Mask Bridge Ex
- **类别：** YogurtNodes/IO
- **描述：** 增强的掩码保存节点，支持多种格式和自定义后缀
- **特点：**
  - 支持PNG/JPEG/自定义格式
  - 可自定义输出目录、文件前缀、后缀
  - 支持元数据、压缩、覆盖保护

#### Save Mask Bridge
- **类别：** YogurtNodes/IO
- **描述：** 基本掩码保存节点
- **特点：**
  - 支持PNG/JPEG格式
  - 可自定义输出目录、文件前缀
  - 支持元数据、压缩、覆盖保护

#### Save Mask Bridge Simple
- **类别：** YogurtNodes/IO
- **描述：** 简化版掩码保存节点
- **特点：**
  - 仅需最基本参数，快速保存掩码

#### Preview Mask Bridge
- **类别：** YogurtNodes/IO
- **描述：** 掩码预览节点
- **特点：**
  - 将掩码保存为临时文件，便于快速预览

### 语言模型节点

#### Gemini Generate Text

- **类别：** YogurtNodes/LLM
- **描述：** 使用Gemini API生成文本
- **特点：**
  - 多种Gemini模型支持
  - 系统提示和用户提示控制
  - 可自定义的生成参数（温度、top_p、top_k等）
  - 安全设置控制
  - 最大输出令牌数设置
  - 自动重试机制

#### Gemini Image Understand

- **类别：** YogurtNodes/LLM
- **描述：** 使用Gemini API理解图像内容
- **特点：**
  - 图像内容分析
  - 与文本提示结合
  - 可自定义的生成参数
  - 多语言支持
  - 详细的图像描述生成

#### Gemini Generate Image

- **类别：** YogurtNodes/LLM
- **描述：** 使用Gemini API生成图片
- **特点：**
  - 支持多种Gemini图像生成模型
  - 系统提示和用户提示控制
  - 可自定义的生成参数（温度、top_p、top_k等）
  - 安全设置控制
  - 最大输出令牌数设置
  - 自动重试机制
  - 输出图片（torch.Tensor）、文本描述和图片数量

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

## 🤝 贡献

欢迎提交PR来帮助改进项目！

## 📄 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件。

## 📞 联系方式

如有问题、bug反馈或功能建议，请[提交Issue](https://github.com/yogurt7771/ComfyUI-YogurtNodes/issues)。

## 🙏 致谢

- ComfyUI社区
- 所有贡献者
