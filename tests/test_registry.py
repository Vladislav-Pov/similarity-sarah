"""Tests for the generic component registry."""

import pytest

from similarity_sarah.registry import Registry


class Base:
    pass


def test_register_and_get_case_insensitive():
    reg: Registry[Base] = Registry("widget")

    @reg.register("Foo")
    class Foo(Base):
        pass

    assert reg.get("foo") is Foo
    assert reg.get("FOO") is Foo
    assert "foo" in reg
    assert reg.names() == ["foo"]


def test_create_instantiates():
    reg: Registry[Base] = Registry("widget")

    @reg.register("bar")
    class Bar(Base):
        def __init__(self, x: int) -> None:
            self.x = x

    obj = reg.create("bar", 5)
    assert isinstance(obj, Bar)
    assert obj.x == 5


def test_multiple_aliases():
    reg: Registry[Base] = Registry("widget")

    @reg.register("a", "b")
    class C(Base):
        pass

    assert reg.get("a") is reg.get("b") is C
    assert reg.names() == ["a", "b"]


def test_duplicate_name_raises():
    reg: Registry[Base] = Registry("widget")

    @reg.register("dup")
    class D(Base):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("dup")
        class E(Base):
            pass


def test_unknown_name_raises():
    reg: Registry[Base] = Registry("widget")
    with pytest.raises(KeyError, match="unknown widget"):
        reg.get("missing")
