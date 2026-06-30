import pytest

from ai_harness.tools.builtins import build_default_registry, calculator, datetime_now, word_count


def test_calculator_handles_parentheses_and_precedence():
    assert calculator("14 * (9 + 3)") == "168"
    assert calculator("125 / 5 + 7") == "32"
    assert calculator("2 + 2 * 3") == "8"


def test_calculator_rejects_arbitrary_code():
    # AST allow-list must reject anything that isn't pure arithmetic.
    with pytest.raises((ValueError, SyntaxError)):
        calculator("__import__('os').system('ls')")
    with pytest.raises((ValueError, SyntaxError)):
        calculator("open('/etc/passwd').read()")


def test_word_count_and_datetime():
    assert word_count("one two three") == "3"
    assert "UTC" in datetime_now("")


def test_registry_runs_tools_and_guards_unknown():
    registry = build_default_registry()
    assert registry.run("calculator", "3 * 3") == "9"
    assert "unknown tool" in registry.run("does_not_exist", "x")


def test_registry_never_raises_on_bad_tool_input():
    registry = build_default_registry()
    out = registry.run("calculator", "__import__('os')")
    assert out.startswith("Error running calculator")
