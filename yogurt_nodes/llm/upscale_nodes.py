"""放大节点：每个模型系列一个节点，输入图片批次逐张处理。"""

from . import protocols  # noqa: F401
from .catalog import upscale
from .framework import ModelFamilyNode


class TopazUpscale(ModelFamilyNode):
    """Topaz Upscale node.

    Topaz precision upscaling (Standard V2, Low Resolution V2, High Fidelity V2, CGI, Text Refine).
    """
    _NODE_NAME = "Topaz Upscale"
    DESCRIPTION = "Topaz precision upscaling (Standard V2, Low Resolution V2, High Fidelity V2, CGI, Text Refine)."
    _SUBCATEGORY = "Upscale"
    FAMILY = upscale.TOPAZ_STANDARD


class TopazGenerativeUpscale(ModelFamilyNode):
    """Topaz Generative Upscale node.

    Topaz generative upscaling (Wonder 3.5, Bloom 2, Reimagine, Redefine, Recovery V2).
    """
    _NODE_NAME = "Topaz Generative Upscale"
    DESCRIPTION = "Topaz generative upscaling (Wonder 3.5, Bloom 2, Reimagine, Redefine, Recovery V2)."
    _SUBCATEGORY = "Upscale"
    FAMILY = upscale.TOPAZ_GENERATIVE


class MagnificCreativeUpscale(ModelFamilyNode):
    """Magnific Creative Upscale node.

    Magnific creative upscaling with prompt, creativity and engine controls.
    """
    _NODE_NAME = "Magnific Creative Upscale"
    DESCRIPTION = "Magnific creative upscaling with prompt, creativity and engine controls."
    _SUBCATEGORY = "Upscale"
    FAMILY = upscale.MAGNIFIC_CREATIVE


class MagnificPrecisionUpscale(ModelFamilyNode):
    """Magnific Precision Upscale node.

    Magnific precision upscaling (V2 with flavors, or V1).
    """
    _NODE_NAME = "Magnific Precision Upscale"
    DESCRIPTION = "Magnific precision upscaling (V2 with flavors, or V1)."
    _SUBCATEGORY = "Upscale"
    FAMILY = upscale.MAGNIFIC_PRECISION
