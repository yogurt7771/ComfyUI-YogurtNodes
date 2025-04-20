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

## 📦 Installation

### Requirements

- ComfyUI (installed and running)
- Python 3.x
- Required Python packages:
  - numpy
  - pillow

### Installation Steps

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/[your-username]/ComfyUI-YogurtNodes.git
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

All nodes are marked with "Yogurt Nodes" for easy identification in the ComfyUI interface.

### Image Processing Nodes

#### Batch Images
- **Category:** YogurtNodes/Image
- **Description:** Combines multiple images into a batch with customizable settings
- **Features:**
  - Support for up to 16 input images
  - Multiple interpolation methods (nearest, bilinear, bicubic, etc.)
  - Various resizing methods (stretch, fill/crop, pad)
  - Customizable padding values
  - Flexible batch slicing with start, end, and step parameters

#### Add Text To Image
- **Category:** YogurtNodes/Image
- **Description:** Adds text overlay to images with extensive customization options
- **Features:**
  - Text placement at top or bottom
  - Center alignment option
  - Multiple font support
  - Customizable font size
  - Text and background color configuration
  - Multi-line text support with automatic wrapping

#### None Image
- **Category:** YogurtNodes/Image
- **Description:** A utility node that returns None as an image output
- **Use Case:** Helpful for conditional workflows where a null image output is needed

### String Processing Nodes

#### String Lines Count
- **Category:** YogurtNodes/String
- **Description:** Counts the number of lines in a multiline string
- **Features:**
  - Optional line stripping
  - Empty line filtering control
  - Configurable counting behavior

#### String To Value
- **Category:** YogurtNodes/String
- **Description:** Converts string input to multiple value types
- **Features:**
  - Automatic conversion to integer
  - Automatic conversion to float
  - Preserves original string
  - Safe conversion with fallback values

#### String Lines Switch
- **Category:** YogurtNodes/String
- **Description:** Extracts a specific line from a multiline string by index
- **Features:**
  - Line indexing
  - Multiple output formats (string, int, float)
  - Line count output
  - Empty line handling options

#### Replace Delimiter
- **Category:** YogurtNodes/String
- **Description:** Replaces delimiters in strings with custom separators
- **Features:**
  - Regular expression support
  - Custom delimiter specification
  - Flexible replacement options

#### Split Path
- **Category:** YogurtNodes/String
- **Description:** Splits file paths into their component parts
- **Features:**
  - Path component extraction
  - Returns parent directory
  - Filename and extension separation
  - Multiple format options for extensions

### Model Selection Nodes

#### ControlNet Selector
- **Category:** YogurtNodes/Models
- **Description:** Advanced ControlNet model selection and configuration
- **Features:**
  - Dynamic model list from available ControlNets
  - Adjustable strength parameter
  - Start and end percentage control
  - Model path and name extraction
  - None option for conditional workflows

#### Diffusion Model Selector
- **Category:** YogurtNodes/Models
- **Description:** Stable Diffusion model selection utility
- **Features:**
  - Dynamic list of available diffusion models
  - Trigger word support
  - Model path and name extraction
  - None option for conditional workflows

#### Checkpoint Selector
- **Category:** YogurtNodes/Models
- **Description:** Checkpoint model selection and management
- **Features:**
  - Dynamic checkpoint list
  - Trigger word support
  - Model information extraction
  - None option for conditional workflows

#### Lora Selector
- **Category:** YogurtNodes/Models
- **Description:** LoRA model selection and configuration
- **Features:**
  - Dynamic LoRA model list
  - Separate model and CLIP strength controls
  - Trigger word support
  - Model path and name extraction
  - None option for conditional workflows

### I/O Operation Nodes

#### Save Image Bridge Ex
- **Category:** YogurtNodes/IO
- **Description:** Enhanced image saving functionality with extensive options
- **Features:**
  - Multiple format support (PNG, JPEG)
  - Customizable output directory
  - Dynamic filename prefixes with variables
  - Metadata control
  - Overwrite protection
  - Compression level control
  - Quality settings for JPEG
  - Custom suffix support

#### Save Image Bridge
- **Category:** YogurtNodes/IO
- **Description:** Basic image saving functionality
- **Features:**
  - PNG and JPEG support
  - Basic output options
  - Metadata handling
  - Compression settings

#### Preview Image Bridge
- **Category:** YogurtNodes/IO
- **Description:** Quick image preview functionality
- **Features:**
  - Temporary file handling
  - Fast preview generation
  - Automatic cleanup

#### Save Text Bridge
- **Category:** YogurtNodes/IO
- **Description:** Text file saving with multiple format support
- **Features:**
  - Multiple format support (.txt, .json, .md, .csv)
  - Custom file extensions
  - Dynamic filename generation
  - Overwrite protection
  - UTF-8 encoding support

#### Any Bridge
- **Category:** YogurtNodes/IO
- **Description:** Universal data bridge for workflow control
- **Features:**
  - Handles any data type
  - Optional blackhole mode
  - Workflow control support

#### Create Directory
- **Category:** YogurtNodes/IO
- **Description:** Directory creation utility
- **Features:**
  - Creates single directory
  - Automatic parent directory creation
  - Path validation

#### Create Parent Directory
- **Category:** YogurtNodes/IO
- **Description:** Parent directory creation utility
- **Features:**
  - Creates parent directories
  - Recursive creation support
  - Path validation
  - Safe operation with existing directories

## 🤝 Contributing

Pull Requests are welcome to help improve the project!

## 📄 License

[Add your license information]

## 📞 Contact

[Add your contact information]

## 🙏 Acknowledgments

- ComfyUI Community
- All Contributors 