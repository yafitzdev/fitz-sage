# tests/test_instrumentation.py
"""
Tests for the instrumentation system.

Tests hook registration, proxy behavior, and caching hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fitz_ai.core.instrumentation import (
    BenchmarkHook,
    InstrumentedProxy,
    _NO_CACHE,
    clear_hooks,
    get_hooks,
    has_hooks,
    maybe_wrap,
    register_hook,
    unregister_hook,
    wrap,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class DummyPlugin:
    """A simple plugin for testing."""

    def __init__(self):
        self.call_count = 0

    def process(self, value: int) -> int:
        """Process a value."""
        self.call_count += 1
        return value * 2

    def fail(self) -> None:
        """Raise an exception."""
        raise ValueError("Intentional failure")

    def _private_method(self) -> str:
        """Private method that should not be instrumented."""
        return "private"


@dataclass(eq=False)
class TrackingHook:
    """Hook that tracks calls for testing.

    Note: eq=False to ensure different instances are not considered equal
    (important for hook registration tests).
    """

    calls: list[dict] = field(default_factory=list)
    end_calls: list[dict] = field(default_factory=list)

    def on_call_start(
        self,
        layer: str,
        plugin_name: str,
        method: str,
        args: tuple,
        kwargs: dict,
    ) -> dict:
        call_info = {
            "layer": layer,
            "plugin": plugin_name,
            "method": method,
            "args": args,
            "kwargs": kwargs,
        }
        self.calls.append(call_info)
        return call_info

    def on_call_end(
        self,
        context: Any,
        result: Any,
        error: Exception | None,
    ) -> None:
        self.end_calls.append({
            "context": context,
            "result": result,
            "error": error,
        })


@dataclass
class CachingHookImpl:
    """Implementation of CachingHook for testing."""

    cache: dict[tuple, Any] = field(default_factory=dict)
    get_calls: int = 0
    set_calls: int = 0

    def on_call_start(self, layer, plugin_name, method, args, kwargs) -> None:
        return None

    def on_call_end(self, context, result, error) -> None:
        pass

    def get_cached_result(
        self,
        layer: str,
        plugin_name: str,
        method: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        self.get_calls += 1
        key = (layer, plugin_name, method, args)
        return self.cache.get(key, _NO_CACHE)

    def cache_result(
        self,
        layer: str,
        plugin_name: str,
        method: str,
        args: tuple,
        kwargs: dict,
        result: Any,
    ) -> None:
        self.set_calls += 1
        key = (layer, plugin_name, method, args)
        self.cache[key] = result


# =============================================================================
# Test Hook Registration
# =============================================================================


class TestHookRegistration:
    """Tests for hook registration."""

    def setup_method(self):
        """Clear hooks before each test."""
        clear_hooks()

    def teardown_method(self):
        """Clear hooks after each test."""
        clear_hooks()

    def test_no_hooks_initially(self):
        """Test that no hooks are registered initially."""
        assert not has_hooks()
        assert get_hooks() == []

    def test_register_hook(self):
        """Test registering a hook."""
        hook = TrackingHook()
        register_hook(hook)

        assert has_hooks()
        assert hook in get_hooks()

    def test_register_same_hook_twice(self):
        """Test that registering the same hook twice is idempotent."""
        hook = TrackingHook()
        register_hook(hook)
        register_hook(hook)

        assert len(get_hooks()) == 1

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        hook = TrackingHook()
        register_hook(hook)
        assert has_hooks()

        unregister_hook(hook)
        assert not has_hooks()

    def test_unregister_nonexistent_hook_raises(self):
        """Test that unregistering a non-existent hook raises ValueError."""
        hook = TrackingHook()

        with pytest.raises(ValueError):
            unregister_hook(hook)

    def test_clear_hooks(self):
        """Test clearing all hooks."""
        # First ensure clean state
        clear_hooks()

        hook1 = TrackingHook()
        hook2 = TrackingHook()
        register_hook(hook1)
        register_hook(hook2)

        assert len(get_hooks()) == 2

        clear_hooks()
        assert not has_hooks()

    def test_get_hooks_returns_copy(self):
        """Test that get_hooks returns a copy, not the internal list."""
        hook = TrackingHook()
        register_hook(hook)

        hooks = get_hooks()
        hooks.clear()  # Modify the returned list

        assert has_hooks()  # Original list should be unchanged


# =============================================================================
# Test InstrumentedProxy
# =============================================================================


class TestInstrumentedProxy:
    """Tests for the InstrumentedProxy class."""

    def setup_method(self):
        """Clear hooks before each test."""
        clear_hooks()

    def teardown_method(self):
        """Clear hooks after each test."""
        clear_hooks()

    def test_proxy_calls_method(self):
        """Test that proxy calls the underlying method."""
        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        result = proxy.process(5)

        assert result == 10
        assert plugin.call_count == 1

    def test_proxy_notifies_hook_on_call(self):
        """Test that proxy notifies hooks on method calls."""
        hook = TrackingHook()
        register_hook(hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test.layer", "dummy_plugin")

        proxy.process(5)

        assert len(hook.calls) == 1
        assert hook.calls[0]["layer"] == "test.layer"
        assert hook.calls[0]["plugin"] == "dummy_plugin"
        assert hook.calls[0]["method"] == "process"
        assert hook.calls[0]["args"] == (5,)

    def test_proxy_notifies_hook_on_end(self):
        """Test that proxy notifies hooks after method completes."""
        hook = TrackingHook()
        register_hook(hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        result = proxy.process(5)

        assert len(hook.end_calls) == 1
        assert hook.end_calls[0]["result"] == 10
        assert hook.end_calls[0]["error"] is None

    def test_proxy_notifies_hook_on_error(self):
        """Test that proxy notifies hooks when method raises exception."""
        hook = TrackingHook()
        register_hook(hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        with pytest.raises(ValueError):
            proxy.fail()

        assert len(hook.end_calls) == 1
        assert hook.end_calls[0]["result"] is None
        assert isinstance(hook.end_calls[0]["error"], ValueError)

    def test_proxy_skips_private_methods(self):
        """Test that proxy does not instrument private methods."""
        hook = TrackingHook()
        register_hook(hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        result = proxy._private_method()

        assert result == "private"
        assert len(hook.calls) == 0  # No hook call for private method

    def test_proxy_respects_methods_to_track(self):
        """Test that proxy only tracks specified methods."""
        hook = TrackingHook()
        register_hook(hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(
            plugin, "test", "dummy",
            methods_to_track={"other_method"}  # Not 'process'
        )

        proxy.process(5)

        assert len(hook.calls) == 0  # 'process' not in methods_to_track

    def test_proxy_fast_path_no_hooks(self):
        """Test that proxy uses fast path when no hooks registered."""
        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        # No hooks registered - should still work
        result = proxy.process(5)
        assert result == 10

    def test_proxy_str_and_repr(self):
        """Test proxy string representations."""
        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test.layer", "dummy")

        assert "InstrumentedProxy" in repr(proxy)
        assert "test.layer" in repr(proxy)
        assert str(plugin) == str(proxy)


# =============================================================================
# Test Caching Hook Integration
# =============================================================================


class TestCachingHookIntegration:
    """Tests for caching hook integration with proxy."""

    def setup_method(self):
        """Clear hooks before each test."""
        clear_hooks()

    def teardown_method(self):
        """Clear hooks after each test."""
        clear_hooks()

    def test_cache_miss_calls_method(self):
        """Test that cache miss results in actual method call."""
        caching_hook = CachingHookImpl()
        register_hook(caching_hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        result = proxy.process(5)

        assert result == 10
        assert plugin.call_count == 1
        assert caching_hook.get_calls == 1
        assert caching_hook.set_calls == 1  # Result was cached

    def test_cache_hit_skips_method(self):
        """Test that cache hit skips actual method call."""
        caching_hook = CachingHookImpl()
        # Pre-populate cache
        caching_hook.cache[("test", "dummy", "process", (5,))] = 99
        register_hook(caching_hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        result = proxy.process(5)

        assert result == 99  # Cached value
        assert plugin.call_count == 0  # Method was NOT called
        assert caching_hook.get_calls == 1

    def test_cache_stores_result(self):
        """Test that result is stored in cache after call."""
        caching_hook = CachingHookImpl()
        register_hook(caching_hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        # First call
        result1 = proxy.process(5)
        assert result1 == 10
        assert plugin.call_count == 1

        # Second call - should use cache
        result2 = proxy.process(5)
        assert result2 == 10
        assert plugin.call_count == 1  # Still 1, method not called again

    def test_no_cache_sentinel(self):
        """Test that _NO_CACHE sentinel is unique."""
        # _NO_CACHE should not equal any normal value
        assert _NO_CACHE is not None
        assert _NO_CACHE != {}
        assert _NO_CACHE != []
        assert _NO_CACHE != ""
        assert _NO_CACHE != 0

    def test_multiple_hooks_both_called(self):
        """Test that multiple hooks are all notified."""
        tracking_hook = TrackingHook()
        caching_hook = CachingHookImpl()
        register_hook(tracking_hook)
        register_hook(caching_hook)

        plugin = DummyPlugin()
        proxy = InstrumentedProxy(plugin, "test", "dummy")

        proxy.process(5)

        assert len(tracking_hook.calls) == 1
        assert caching_hook.get_calls == 1


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for maybe_wrap and wrap functions."""

    def setup_method(self):
        """Clear hooks before each test."""
        clear_hooks()

    def teardown_method(self):
        """Clear hooks after each test."""
        clear_hooks()

    def test_maybe_wrap_no_hooks(self):
        """Test that maybe_wrap returns original when no hooks."""
        plugin = DummyPlugin()

        result = maybe_wrap(plugin, "test", "dummy")

        assert result is plugin  # Same object, not wrapped

    def test_maybe_wrap_with_hooks(self):
        """Test that maybe_wrap returns proxy when hooks registered."""
        hook = TrackingHook()
        register_hook(hook)

        plugin = DummyPlugin()

        result = maybe_wrap(plugin, "test", "dummy")

        assert isinstance(result, InstrumentedProxy)
        assert result is not plugin

    def test_wrap_always_wraps(self):
        """Test that wrap always returns a proxy."""
        plugin = DummyPlugin()

        result = wrap(plugin, "test", "dummy")

        assert isinstance(result, InstrumentedProxy)

    def test_wrap_with_methods_to_track(self):
        """Test wrap with specific methods to track."""
        plugin = DummyPlugin()

        result = wrap(plugin, "test", "dummy", methods_to_track={"process"})

        assert isinstance(result, InstrumentedProxy)


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of hook registration."""

    def setup_method(self):
        """Clear hooks before each test."""
        clear_hooks()

    def teardown_method(self):
        """Clear hooks after each test."""
        clear_hooks()

    def test_concurrent_registration(self):
        """Test that concurrent registration is safe."""
        import threading

        # Ensure clean state first
        clear_hooks()

        hooks = [TrackingHook() for _ in range(100)]
        threads = []

        def register_hooks(start, end):
            for i in range(start, end):
                register_hook(hooks[i])

        # Register from multiple threads
        for i in range(10):
            t = threading.Thread(target=register_hooks, args=(i * 10, (i + 1) * 10))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All unique hooks should be registered
        assert len(get_hooks()) == 100
