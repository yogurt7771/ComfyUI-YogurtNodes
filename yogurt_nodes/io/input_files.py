import os

import folder_paths


def list_input_files(content_types):
    """Relative paths of input files (recursive) matching the given content types, sorted."""
    input_dir = folder_paths.get_input_directory()
    # os.walk takes file/dir types from the directory listing. Path.rglob + is_file()
    # stats every entry, which takes ~30 s per node on NFS input folders with 100k+ files.
    files = [
        os.path.relpath(os.path.join(root, name), input_dir).replace(os.sep, "/")
        for root, _, names in os.walk(input_dir)
        for name in names
        if "." in name
    ]
    return sorted(folder_paths.filter_files_content_types(files, content_types))
