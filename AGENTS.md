# Project Agent Instructions

- For Python commands in this project, use the ComfyUI virtual environment:
  `I:\Codes\AIArt\ComfyUI\venv\python.exe`.
- This copied environment is conda/conda-forge style. Do not use
  `I:\Codes\AIArt\ComfyUI\venv\Scripts\python.exe` for tests, formatting,
  compile checks, or project scripts, because it can fail to load environment
  DLL dependencies such as `_ctypes` and `_ssl`.
- Do not copy or overwrite `venv\python.exe` onto `venv\Scripts\python.exe`;
  use the root `venv\python.exe` entry point instead.
- Do not use the global `python` executable or a local `.\venv` for tests,
  formatting, compile checks, or project scripts unless the user explicitly asks
  for it.
- Add `I:\Codes\AIArt\ComfyUI` to `sys.path` for project Python runs. For
  shell commands, set `PYTHONPATH` before invoking Python, for example:
  `$env:PYTHONPATH = "I:\Codes\AIArt\ComfyUI;$env:PYTHONPATH"`.
- Example verification commands:
  - `& "I:\Codes\AIArt\ComfyUI\venv\python.exe" -m unittest discover -s tests -v`
  - `& "I:\Codes\AIArt\ComfyUI\venv\python.exe" -m compileall yogurt_nodes tests`
- `venv/` is ignored by git. Do not read files inside ignored virtual environment
  directories unless the user explicitly grants permission.
