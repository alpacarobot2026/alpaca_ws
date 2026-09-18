from pathlib import Path

def get_src_project_dir_path(src_proj_path: str) -> Path:
    # logger.debug(f"debug:{__file__}\n{Path(__file__).parents}")
    for parent in Path(__file__).parents:
        if (parent / src_proj_path).exists():
            # parent is workspace directory
            return parent / src_proj_path
        
    assert Exception(f"Cannot find the source code path {src_proj_path}")