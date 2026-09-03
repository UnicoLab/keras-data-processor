"""Every constructor argument has to be read, or say that it is not.

An argument a class stores and never reads is the quietest kind of bug: the
call is accepted, the attribute is there when you inspect it, `get_config`
round-trips it, and the behaviour it names never happens. This engagement found
eight of them -- `feature_moe_use_residual`, `use_embedding`,
`cyclic_encoding`, `pad_value`, `onehot_categorical`, `drop_na` on the wavelet,
`handle_sparsity`, and `extrapolate`, which let a `NaN` through to the model.

So: an argument stored on `self` and read nowhere outside `get_config` must say
so in the class docstring. Implement it, or write down that it does nothing.
"""

import ast
import re
import unittest
from pathlib import Path

import pytest

KDP = Path(__file__).resolve().parent.parent / "kdp"

# The phrasing that admits an argument does nothing. Whichever a docstring
# uses, a reader learns the truth before writing code against it.
DISCLAIMERS = (
    "not used",
    "accepted and not used",
    "no effect",
    "deprecated and ignored",
    "ignored",
)


def _stored_parameters(init: ast.FunctionDef, params: set[str]) -> dict[str, str]:
    """Attributes assigned straight from a constructor argument."""
    stored = {}
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                sources = {
                    name.id
                    for name in ast.walk(node.value)
                    if isinstance(name, ast.Name)
                } & params
                if sources:
                    stored[target.attr] = sorted(sources)[0]
    return stored


def _package_attribute_reads() -> set[str]:
    """Attribute names read anywhere in the package, on any object.

    A feature class stores what the processor reads later:
    `NumericalFeature.use_embedding` is set in `kdp/features.py` and read in
    `kdp/processor.py`, sometimes through `getattr(feature, "use_embedding")`.
    Scanning one class alone would call that dropped.
    """
    names = set()
    for path in KDP.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                # `self.x` inside the defining class is handled separately; a
                # read through any other object counts here.
                if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                    names.add(node.attr)
            # getattr(obj, "name") and hasattr(obj, "name").
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "hasattr"}
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                names.add(node.args[1].value)
    return names


def _subclass_self_reads() -> dict[str, set[str]]:
    """For each class name, the `self.x` reads made by classes deriving from it.

    A base class can legitimately store what only its subclasses read --
    `InferenceFormatter` holds the preprocessor that its time series subclass
    queries -- so those reads have to count for the base.
    """
    own: dict[str, set[str]] = {}
    bases: dict[str, list[str]] = {}
    for path in KDP.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            own.setdefault(cls.name, set()).update(_attributes_read(cls))
            bases.setdefault(cls.name, []).extend(
                base.id for base in cls.bases if isinstance(base, ast.Name)
            )

    inherited: dict[str, set[str]] = {name: set() for name in own}
    for name in own:
        # Walk up from every class, crediting each ancestor with its reads.
        seen = set()
        stack = list(bases.get(name, []))
        while stack:
            parent = stack.pop()
            if parent in seen:
                continue
            seen.add(parent)
            if parent in inherited:
                inherited[parent] |= own[name]
            stack.extend(bases.get(parent, []))
    return inherited


def _attributes_read(cls: ast.ClassDef) -> set[str]:
    """Every `self.x` read in the class, except inside `get_config`."""
    read = set()
    for member in cls.body:
        if isinstance(member, ast.FunctionDef) and member.name == "get_config":
            continue
        for node in ast.walk(member):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Load)
            ):
                read.add(node.attr)
    return read


def _init_reads(init: ast.FunctionDef) -> set[str]:
    """`self.x` read back inside `__init__` -- validation, derived values."""
    return {
        node.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
    }


def _locals_consumed(init: ast.FunctionDef, params: set[str]) -> set[str]:
    """Arguments used somewhere in `__init__` beyond a bare `self.x = x`.

    A layer that builds a sublayer with `MultiHeadAttention(num_heads=num_heads)`
    honours the argument; the matching `self.num_heads` is only there so
    `get_config` can round-trip it.
    """
    consumed = set()
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            targets = [
                t
                for t in node.targets
                if isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ]
            if targets and isinstance(node.value, ast.Name):
                # A plain `self.x = x` passes the argument along, nothing more.
                continue
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name) and child.id in params:
                consumed.add(child.id)
    return consumed


def _unread_arguments() -> list[tuple[str, str, str]]:
    """Every (file, class, attribute) stored from an argument and never read."""
    findings = []
    elsewhere = _package_attribute_reads()
    from_subclasses = _subclass_self_reads()
    for path in sorted(KDP.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            init = next(
                (
                    n
                    for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            params = {arg.arg for arg in init.args.args if arg.arg != "self"}
            params |= {arg.arg for arg in init.args.kwonlyargs}
            if not params:
                continue
            stored = _stored_parameters(init, params)
            read = (
                _attributes_read(cls)
                | _init_reads(init)
                | from_subclasses.get(cls.name, set())
            )
            consumed = _locals_consumed(init, params)
            docstring = (ast.get_docstring(cls) or "") + (ast.get_docstring(init) or "")
            for attribute, argument in sorted(stored.items()):
                if attribute in read or argument in consumed:
                    continue
                if attribute in elsewhere:
                    continue
                if _is_disclaimed(docstring, argument):
                    continue
                findings.append(
                    (str(path.relative_to(KDP.parent)), cls.name, argument),
                )
    return findings


def _is_disclaimed(docstring: str, argument: str) -> bool:
    """Does the docstring say, next to this argument, that it does nothing?"""
    entry_start = re.compile(
        rf"^\s*\*{{0,2}}{re.escape(argument)}\b\s*(\([^)]*\))?\s*:"
    )
    lines = docstring.split("\n")
    for index, line in enumerate(lines):
        if not entry_start.match(line):
            continue
        # The entry plus its indented continuation lines.
        entry = [line]
        for follow in lines[index + 1 :]:
            if follow.strip() and not follow.startswith(" " * 12):
                break
            entry.append(follow)
        text = " ".join(entry).lower()
        if any(phrase in text for phrase in DISCLAIMERS):
            return True
    return False


@pytest.mark.unit
class TestEveryOptionIsReadOrDisclaimed(unittest.TestCase):
    """The guard itself."""

    def test_no_constructor_argument_is_quietly_dropped(self):
        findings = _unread_arguments()
        message = "\n".join(
            f"  {path}: {cls}({argument}=...) is stored and never read. "
            "Implement it, or say so in the class docstring."
            for path, cls, argument in findings
        )
        self.assertEqual(findings, [], f"\n{message}")

    def test_the_guard_notices_a_dropped_argument(self):
        """A guard that cannot fail proves nothing."""
        source = '''
class Example:
    """A class.

    Args:
        used: A real one.
        dropped: Sounds important.
    """

    def __init__(self, used, dropped):
        self.used = used
        self.dropped = dropped

    def call(self, x):
        return x * self.used
'''
        tree = ast.parse(source)
        cls = tree.body[0]
        init = cls.body[1]
        params = {"used", "dropped"}
        stored = _stored_parameters(init, params)
        read = _attributes_read(cls) | _init_reads(init)
        consumed = _locals_consumed(init, params)
        missed = [
            argument
            for attribute, argument in stored.items()
            if attribute not in read and argument not in consumed
        ]
        self.assertEqual(missed, ["dropped"])
        self.assertFalse(_is_disclaimed(ast.get_docstring(cls), "dropped"))
