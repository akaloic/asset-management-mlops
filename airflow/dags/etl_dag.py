"""
Cleanup script to remove empty folders and unused files.
"""
import os
import shutil
from pathlib import Path


def remove_empty_folders(root_dir):
    """Remove all empty folders recursively."""
    removed = []
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            try:
                if not os.listdir(full_path):
                    os.rmdir(full_path)
                    removed.append(full_path)
                    print(f"Removed empty folder: {full_path}")
            except OSError:
                pass
    return removed


def remove_cache_files(root_dir):
    """Remove Python cache and temporary files."""
    patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.pytest_cache',
        '.ipynb_checkpoints',
        '*.log',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    for pattern in patterns:
        if '*' in pattern:
            for path in Path(root_dir).rglob(pattern):
                try:
                    path.unlink()
                    print(f"Removed file: {path}")
                except:
                    pass
        else:
            for path in Path(root_dir).rglob(pattern):
                try:
                    shutil.rmtree(path)
                    print(f"Removed cache: {path}")
                except:
                    pass


def remove_empty_files(root_dir):
    """Remove files with 0 bytes."""
    for path in Path(root_dir).rglob('*'):
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
            print(f"Removed empty file: {path}")


if __name__ == "__main__":
    project_root = Path(__file__).parent
    
    print("Starting project cleanup...")
    print("=" * 60)
    
    remove_cache_files(project_root)
    remove_empty_files(project_root)
    remove_empty_folders(project_root)
    
    print("=" * 60)
    print("Cleanup completed!")
