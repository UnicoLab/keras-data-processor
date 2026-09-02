"""Guard against the documentation drifting away from the code.

Every defect this catches was present in the published docs: keyword arguments
that raised `TypeError`, imports of classes that were never written, feature
types that do not exist, and option strings that matched nothing and therefore
silently disabled the feature they were supposed to configure.

The feature classes accept `**kwargs`, so a wrong keyword there is not an
error at call time -- it is simply ignored. That is why an unknown keyword is
treated as a defect unless the source reads it or it is forwarded to the Keras
layer that ultimately receives it.
"""

import ast
import importlib
import sys
import inspect
import re
import subprocess
import unittest
from enum import Enum
from pathlib import Path

import keras
import pytest

import kdp
import kdp.features
import kdp.processor

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
FENCE = re.compile(r"```python\n(.*?)```", re.S)

FEATURE_CLASSES = {
    "Feature",
    "NumericalFeature",
    "CategoricalFeature",
    "TextFeature",
    "DateFeature",
    "TimeSeriesFeature",
    "PassthroughFeature",
}

# Options whose value is compared by string equality, so a near-miss is fatal.
OPTION_PARAMS = {
    "feature_selection_placement": kdp.processor.FeatureSelectionPlacementOptions,
    "tabular_attention_placement": kdp.processor.TabularAttentionPlacementOptions,
    "transfo_placement": kdp.processor.TransformerBlockPlacementOptions,
    "output_mode": kdp.processor.OutputModeOptions,
}


def _option_values(options) -> set[str]:
    """Collect the valid string values a placement/mode option accepts."""
    return {
        (v.value if isinstance(v, Enum) else v)
        for k, v in vars(options).items()
        if not k.startswith("_") and isinstance(v, str | Enum)
    }


def _documentation_files():
    """Every hand-written markdown file, the README included.

    `docs/generated/` is produced from docstrings by a script, so it is checked
    at the source rather than here.
    """
    files = [p for p in sorted(DOCS.rglob("*.md")) if "generated" not in p.parts]
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        files.append(readme)
    return files


def _doc_blocks():
    """Yield (path, index, source) for every python block in the docs."""
    for md in _documentation_files():
        try:
            relative = md.relative_to(DOCS)
        except ValueError:
            relative = md.relative_to(REPO_ROOT)
        for index, code in enumerate(FENCE.findall(md.read_text())):
            yield relative, index, code


def _identifiers_read_by_the_source() -> set[str]:
    """Every lowercase identifier that appears anywhere under kdp/.

    A keyword argument the package never mentions cannot be doing anything.
    """
    result = subprocess.run(
        ["grep", "-rhoE", r"[a-z_]{3,}", "--include=*.py", str(REPO_ROOT / "kdp")],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(result.stdout.split())


# Keyword arguments forwarded verbatim to the Keras layer that consumes them.
FORWARDED = set(inspect.signature(keras.layers.TextVectorization.__init__).parameters)

CONSUMED = _identifiers_read_by_the_source() | FORWARDED

FEATURE_TYPES = {m.name for m in kdp.features.FeatureType}
PM_PARAMS = set(inspect.signature(kdp.PreprocessingModel.__init__).parameters) - {
    "self"
}


@pytest.mark.unit
class TestDocumentedCodeMatchesTheAPI(unittest.TestCase):
    """The documentation is checked against the package it documents."""

    def test_every_python_block_parses(self):
        """A block that does not parse cannot have been run by its author."""
        broken = []
        for path, index, code in _doc_blocks():
            try:
                ast.parse(code)
            except SyntaxError as exc:
                # Indented fragments are excerpts, not runnable programs.
                if "unexpected indent" in str(exc.msg):
                    continue
                broken.append(f"{path} block #{index}: {exc.msg} (line {exc.lineno})")
        self.assertEqual(broken, [], "\n".join(broken))

    def test_imported_names_exist(self):
        """`from kdp... import X` must resolve."""
        missing = []
        for path, index, code in _doc_blocks():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith("kdp"):
                    continue
                try:
                    module = importlib.import_module(node.module)
                except ImportError:
                    missing.append(f"{path} block #{index}: no module {node.module}")
                    continue
                for alias in node.names:
                    if alias.name != "*" and not hasattr(module, alias.name):
                        missing.append(
                            f"{path} block #{index}: {node.module} has no {alias.name}"
                        )
        self.assertEqual(missing, [], "\n".join(missing))

    def test_feature_types_exist(self):
        """`FeatureType.X` must be a real member."""
        missing = []
        for path, index, code in _doc_blocks():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "FeatureType"
                    and node.attr not in FEATURE_TYPES
                ):
                    missing.append(f"{path} block #{index}: FeatureType.{node.attr}")
        self.assertEqual(missing, [], "\n".join(missing))

    def test_preprocessing_model_kwargs_exist(self):
        """`PreprocessingModel` has no `**kwargs`, so these raise TypeError."""
        unknown = []
        for path, index, code in _doc_blocks():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "PreprocessingModel"
                ):
                    for keyword in node.keywords:
                        if keyword.arg and keyword.arg not in PM_PARAMS:
                            unknown.append(f"{path} block #{index}: {keyword.arg}")
        self.assertEqual(unknown, [], "\n".join(unknown))

    def test_feature_kwargs_are_read_somewhere(self):
        """Feature classes swallow `**kwargs`, so a typo changes nothing."""
        ignored = []
        for path, index, code in _doc_blocks():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in FEATURE_CLASSES
                ):
                    for keyword in node.keywords:
                        if keyword.arg and keyword.arg not in CONSUMED:
                            ignored.append(
                                f"{path} block #{index}: "
                                f"{node.func.id}({keyword.arg}=...)"
                            )
        self.assertEqual(ignored, [], "\n".join(ignored))

    def test_option_strings_are_valid(self):
        """A near-miss silently disables the feature rather than raising."""
        invalid = []
        for path, index, code in _doc_blocks():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                # `output_mode` means different things on PreprocessingModel
                # (concat/dict) and on TextFeature (int/multi_hot/count), so
                # only check it where it names the model-level option.
                if not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "PreprocessingModel":
                    continue
                for keyword in node.keywords:
                    options = OPTION_PARAMS.get(keyword.arg)
                    if options is None:
                        continue
                    if not isinstance(keyword.value, ast.Constant):
                        continue
                    value = keyword.value.value
                    if not isinstance(value, str):
                        continue
                    valid = _option_values(options)
                    if not any(value.lower() == v.lower() for v in valid):
                        invalid.append(
                            f"{path} block #{index}: {keyword.arg}={value!r} "
                            f"not in {sorted(valid)}"
                        )
        self.assertEqual(invalid, [], "\n".join(invalid))

    def test_every_module_reaches_the_api_reference(self):
        """A module the generator does not list has no API page at all.

        `kdp.moe` and `kdp.model_advisor` were both absent, so the whole
        Feature-MoE implementation and the advisor behind `auto_configure` were
        missing from the published reference while being documented as
        headline features elsewhere.
        """
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            generator = importlib.import_module("generate_docstring_docs")
        finally:
            sys.path.pop(0)

        listed = set(generator.MODULES_TO_DOCUMENT)
        package_root = Path(kdp.__file__).parent
        missing = []
        for path in sorted(package_root.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            module = "kdp." + ".".join(
                path.relative_to(package_root).with_suffix("").parts
            )
            if module in listed:
                continue
            public = [
                node.name
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
            ]
            if public:
                missing.append(f"{module}: {public}")
        self.assertEqual(missing, [], "\n".join(missing))

    def test_internal_links_resolve(self):
        """Every cross-link between pages must point at a page that exists.

        Ten of them did not, including the homepage's own link to Feature
        Selection. `mkdocs build --strict` only checks markdown links, and the
        docs navigate with raw `<a href>` inside HTML blocks, so the dead ones
        rendered as links that go nowhere.
        """
        href = re.compile(r'href="([^"]+)"')
        markdown_link = re.compile(r"\]\(([^)\s]+)\)")
        # Links written inside code spans are examples, not navigation.
        code_span = re.compile(r"`[^`]*`")

        broken = []
        for path in sorted(DOCS.rglob("*.md")):
            if "generated" in path.parts:
                continue
            text = code_span.sub("", path.read_text())
            for link in href.findall(text) + markdown_link.findall(text):
                if link.startswith(("http://", "https://", "#", "mailto:", "data:")):
                    continue
                target = (path.parent / link.split("#")[0]).resolve()
                if not target.exists():
                    broken.append(f"{path.relative_to(DOCS)} -> {link}")
        self.assertEqual(broken, [], "\n".join(broken))

    def test_documented_methods_exist(self):
        """A method the docs call must be a method the class has.

        The documentation showed `get_feature_importance`,
        `plot_feature_importance`, `get_top_features`, `update_statistics`,
        `transform` and `fit` on a `PreprocessingModel`. None of them existed:
        following those pages raised `AttributeError`.
        """
        # Names bound to a PreprocessingModel in the examples, plus the ones
        # the prose uses without showing the assignment.
        implicit = {"preprocessor", "preprocessing_model", "basic", "advanced"}
        # `dir` misses attributes assigned on instances, such as the built
        # `model`, which the examples legitimately call.
        source = ast.parse(Path(kdp.processor.__file__).read_text())
        model_class = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == "PreprocessingModel"
        )
        assigned = {
            target.attr
            for node in ast.walk(model_class)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        real = set(dir(kdp.processor.PreprocessingModel)) | assigned
        phantom = []
        for path, index, code in _doc_blocks():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            names = set(implicit)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    function = node.value.func
                    if isinstance(function, ast.Name) and function.id == (
                        "PreprocessingModel"
                    ):
                        names.update(
                            target.id
                            for target in node.targets
                            if isinstance(target, ast.Name)
                        )
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in names
                    and node.func.attr not in real
                ):
                    phantom.append(
                        f"{path} block #{index}: PreprocessingModel has no "
                        f"{node.func.attr}()"
                    )
        self.assertEqual(sorted(set(phantom)), [], "\n".join(sorted(set(phantom))))

    def test_every_constructor_parameter_is_documented(self):
        """A parameter no page mentions is a capability nobody can find.

        Five options reached users this way: the whole predefined-routing
        control surface of the feature mixture, plus file logging. They worked,
        they were reachable, and the prose never named them.
        """
        prose = "\n".join(
            path.read_text()
            for path in sorted(DOCS.rglob("*.md"))
            # The API pages are generated from the docstrings, so they always
            # match by construction and would mask a gap in the written docs.
            if "generated" not in path.parts
        )
        undocumented = sorted(param for param in PM_PARAMS if param not in prose)
        self.assertEqual(
            undocumented,
            [],
            "PreprocessingModel parameters missing from the documentation: "
            + ", ".join(undocumented),
        )


if __name__ == "__main__":
    unittest.main()
