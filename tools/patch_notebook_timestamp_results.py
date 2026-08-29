from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS_DIR = Path(r"C:\Users\bhara\Desktop\LLM\MosaicFormer\NOTEBOOKS")


def replace_source(cell: dict, old: str, new: str) -> bool:
    source = cell.get("source")
    if not isinstance(source, list):
        return False
    text = "".join(source)
    if old not in text:
        return False
    text = text.replace(old, new)
    cell["source"] = text.splitlines(keepends=True)
    return True


def patch_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        changed |= replace_source(
            cell,
            "import math\nimport pandas as pd",
            "import math\nfrom datetime import datetime\nimport pandas as pd",
        )
        changed |= replace_source(
            cell,
            "comparison.to_csv(RESULT_DIR / 'experiment_results_2.csv', index=False)",
            "RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')\n"
            "comparison.to_csv(RESULT_DIR / f'experiment_results_2_{RUN_TIMESTAMP}.csv', index=False)",
        )
        changed |= replace_source(
            cell,
            "plt.savefig(RESULT_DIR / 'mosaicformer_experiment_loss_2.png', dpi=150)",
            "plt.savefig(RESULT_DIR / f'mosaicformer_experiment_loss_2_{RUN_TIMESTAMP}.png', dpi=150)",
        )

    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"patched {path.name}")


def main() -> None:
    for path in NOTEBOOKS_DIR.glob("*.ipynb"):
        patch_notebook(path)


if __name__ == "__main__":
    main()
