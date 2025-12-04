import json
import sys
import shutil
from pathlib import Path


def clean_notebook(path: Path):
    # Backup original
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)

    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    metadata = nb.get("metadata", {})
    removed = {
        "top_level_widgets": False,
        "cell_outputs_removed": 0,
    }

    # Remove top-level widget state if present
    if "widgets" in metadata:
        metadata.pop("widgets", None)
        nb["metadata"] = metadata
        removed["top_level_widgets"] = True

    # Remove outputs that contain widget MIME types
    widget_prefix = "application/vnd.jupyter.widget"
    for cell in nb.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        new_outputs = []
        for out in outputs:
            data = out.get("data", {})
            # If any data key begins with widget mime, skip this output
            if any(k.startswith(widget_prefix) for k in data.keys()):
                removed["cell_outputs_removed"] += 1
                continue
            new_outputs.append(out)
        if len(new_outputs) != len(outputs):
            cell["outputs"] = new_outputs

    # Write cleaned notebook back
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Backup written to: {backup_path}")
    print(f"Top-level widgets removed: {removed['top_level_widgets']}")
    print(f"Cell outputs removed (widget MIME): {removed['cell_outputs_removed']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_notebook_widgets.py <notebook.ipynb>")
        sys.exit(1)

    nb_path = Path(sys.argv[1])
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        sys.exit(1)

    clean_notebook(nb_path)
