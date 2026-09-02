"""Execute the self-contained examples in the documentation.

The AST guard proves every symbol exists. It cannot prove the example works:
a wrong shape, an option combination Keras rejects, or a method that returns
something other than what the surrounding prose claims all pass a symbol check
and fail the moment someone runs the code.

Each candidate block is run in its own process against a CSV synthesised from
the feature spec the block itself declares.
"""

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
FENCE = re.compile(r"```python\n(.*?)```", re.S)
PYTHON = sys.executable

# Column generators keyed by the FeatureType the block asks for.
GENERATORS = {
    "FLOAT": "rng.normal(50, 10, N)",
    "FLOAT_NORMALIZED": "rng.normal(50, 10, N)",
    "FLOAT_RESCALED": "rng.normal(50, 10, N)",
    "FLOAT_DISCRETIZED": "rng.normal(50, 10, N)",
    "INTEGER_CATEGORICAL": "rng.integers(0, 8, N)",
    "STRING_CATEGORICAL": 'rng.choice(["alpha", "beta", "gamma"], N)',
    "TEXT": 'rng.choice(["hello world data", "science rocks here"], N)',
    "DATE": 'pd.date_range("2021-01-01", periods=N).strftime("%Y-%m-%d")',
    "TIME_SERIES": "np.linspace(1, 100, N)",
    "PASSTHROUGH": "rng.normal(1, 1, N)",
}
DEFAULT_GEN = "rng.normal(50, 10, N)"


def feature_columns(code: str) -> dict[str, str]:
    """Infer the columns a block needs from its features_specs literal."""
    columns: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return columns

    def record(key: str, value_node) -> None:
        kind = None
        # A spec may be the plain string name of a feature type.
        if (
            isinstance(value_node, ast.Constant)
            and isinstance(value_node.value, str)
            and value_node.value in GENERATORS
        ):
            columns[key] = GENERATORS[value_node.value]
            return
        # An explicit dtype= on a passthrough decides the column type.
        for node in ast.walk(value_node):
            if isinstance(node, ast.keyword) and node.arg == "dtype":
                text = ast.unparse(node.value)
                if "string" in text:
                    columns[key] = GENERATORS["STRING_CATEGORICAL"]
                    return
                if "int" in text:
                    columns[key] = "rng.integers(1000, 1100, N)"
                    return
        for node in ast.walk(value_node):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "FeatureType"
            ):
                kind = node.attr
                break
            if isinstance(node, ast.Name) and node.id.endswith("Feature"):
                kind = {
                    "NumericalFeature": "FLOAT_NORMALIZED",
                    "CategoricalFeature": "INTEGER_CATEGORICAL",
                    "TextFeature": "TEXT",
                    "DateFeature": "DATE",
                    "TimeSeriesFeature": "TIME_SERIES",
                    "PassthroughFeature": "PASSTHROUGH",
                }.get(node.id)
                if kind:
                    break
        # A later dict in the same block (a prediction batch, say) must not
        # overwrite a column whose type we already inferred from the spec.
        if kind is None and key in columns:
            return
        columns[key] = GENERATORS.get(kind, DEFAULT_GEN)

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=False):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    record(k.value, v)
        # sort_by / group_by reference extra columns
        if (
            isinstance(node, ast.keyword)
            and node.arg in {"sort_by", "group_by"}
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            name = node.value.value
            columns.setdefault(
                name,
                GENERATORS["DATE"]
                if "date" in name or "time" in name
                else GENERATORS["STRING_CATEGORICAL"],
            )
    return columns


def csv_paths(code: str) -> list[str]:
    """Every string literal in the block that looks like a CSV path."""
    return sorted(
        set(re.findall(r'"([^"]+\.csv)"', code))
        | set(re.findall(r"'([^']+\.csv)'", code)),
    )


def is_candidate(code: str) -> bool:
    """A block worth running: it builds a preprocessor from a spec."""
    if "PreprocessingModel(" not in code:
        return False
    if "features_specs" not in code:
        return False
    # Blocks that train a downstream model or read files we cannot invent.
    skip = ("my_model", "model.fit(", "load_model(", "read_csv", "...")
    return all(marker not in code for marker in skip)


def harness(code: str, columns: dict[str, str], paths: list[str]) -> str:
    """Wrap a block so it runs against synthesised data."""
    cols = ",\n        ".join(f'"{k}": {v}' for k, v in columns.items())
    writes = "\n".join(
        f'    _p = _base / "{Path(p).name}"\n'
        f"    _p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"    _frame.to_csv(_p, index=False)\n"
        f'    _paths["{p}"] = str(_p)'
        for p in paths
    )
    return f"""import os, sys, tempfile, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np, pandas as pd
import keras, tensorflow as tf
from loguru import logger
logger.remove()

N = 150
rng = np.random.default_rng(0)
_frame = pd.DataFrame({{
        {cols}
}})
_paths = {{}}
with tempfile.TemporaryDirectory() as _tmp:
    _base = Path(_tmp)
{writes if writes else "    pass"}

    # Run from the temporary directory. A block that names a relative path --
    # `features_stats_path="features_stats.json"`, say -- would otherwise write
    # it into the repository, and a stray stats file there makes the tests that
    # assert "no data means a clear error" find statistics and fail.
    os.chdir(_tmp)

    # Point every documented path at the synthesised copy.
    _orig_init = None
    import kdp
    _RealModel = kdp.PreprocessingModel
    class _Patched(_RealModel):
        def __init__(self, *a, **kw):
            if "path_data" in kw and isinstance(kw["path_data"], str):
                kw["path_data"] = _paths.get(kw["path_data"], kw["path_data"])
            elif "path_data" not in kw:
                kw["path_data"] = next(iter(_paths.values()), None)
            kw.setdefault("features_stats_path", str(_base / "stats.json"))
            kw.setdefault("overwrite_stats", True)
            super().__init__(*a, **kw)
    import kdp.processor
    kdp.PreprocessingModel = _Patched
    kdp.processor.PreprocessingModel = _Patched
    PreprocessingModel = _Patched

    keras.backend.clear_session()
{chr(10).join("    " + line for line in code.split(chr(10)))}
print("__BLOCK_OK__")
"""


def main() -> None:
    results = []
    for md in sorted(DOCS.rglob("*.md")):
        if "generated" in md.parts:
            continue
        for index, code in enumerate(FENCE.findall(md.read_text())):
            if not is_candidate(code):
                continue
            cols = feature_columns(code)
            if not cols:
                continue
            script = harness(code, cols, csv_paths(code) or ["data.csv"])
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                fh.write(script)
                script_path = fh.name
            proc = subprocess.run(
                [PYTHON, script_path],
                capture_output=True,
                text=True,
                timeout=600,
            )
            ok = "__BLOCK_OK__" in proc.stdout
            results.append(
                {
                    "file": str(md.relative_to(DOCS)),
                    "block": index,
                    "ok": ok,
                    "error": "" if ok else (proc.stderr.strip().split("\n")[-1][:200]),
                },
            )
            Path(script_path).unlink(missing_ok=True)

    failed = [r for r in results if not r["ok"]]
    print(f"ran {len(results)} runnable doc blocks; {len(failed)} failed\n")
    for r in failed:
        print(f"  {r['file']} block #{r['block']}")
        print(f"      {r['error']}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
