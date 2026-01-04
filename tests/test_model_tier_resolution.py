# tests/test_model_tier_resolution.py
"""
Tests for model tier resolution logic.

Tests the three-tier system (smart, fast, balanced) and fallback behavior.
"""

from __future__ import annotations

import warnings

from fitz_ai.llm.runtime import _resolve_model_from_tier


class TestModelTierResolution:
    """Tests for _resolve_model_from_tier function."""

    def test_user_model_takes_priority(self):
        """Test that user-specified model overrides tier logic."""
        defaults = {"models": {"smart": "gpt-4o", "fast": "gpt-4o-mini"}}
        user_kwargs = {"model": "custom-model"}

        result = _resolve_model_from_tier(
            defaults, tier="smart", user_kwargs=user_kwargs, plugin_name="test"
        )

        # User model specified, so None returned (no tier resolution)
        assert result is None

    def test_smart_tier_returns_smart_model(self):
        """Test that smart tier returns the smart model."""
        defaults = {"models": {"smart": "gpt-4o", "fast": "gpt-4o-mini", "balanced": "gpt-4o-mini"}}

        result = _resolve_model_from_tier(
            defaults, tier="smart", user_kwargs={}, plugin_name="test"
        )

        assert result == "gpt-4o"

    def test_fast_tier_returns_fast_model(self):
        """Test that fast tier returns the fast model."""
        defaults = {"models": {"smart": "gpt-4o", "fast": "gpt-4o-mini", "balanced": "gpt-4o-mini"}}

        result = _resolve_model_from_tier(defaults, tier="fast", user_kwargs={}, plugin_name="test")

        assert result == "gpt-4o-mini"

    def test_balanced_tier_returns_balanced_model(self):
        """Test that balanced tier returns the balanced model."""
        defaults = {"models": {"smart": "gpt-4o", "fast": "gpt-4o-mini", "balanced": "gpt-4o-mini"}}

        result = _resolve_model_from_tier(
            defaults, tier="balanced", user_kwargs={}, plugin_name="test"
        )

        assert result == "gpt-4o-mini"

    def test_default_tier_is_smart(self):
        """Test that None tier defaults to smart."""
        defaults = {"models": {"smart": "gpt-4o", "fast": "gpt-4o-mini"}}

        result = _resolve_model_from_tier(defaults, tier=None, user_kwargs={}, plugin_name="test")

        assert result == "gpt-4o"

    # -------------------------------------------------------------------------
    # Fallback tests
    # -------------------------------------------------------------------------

    def test_smart_fallback_to_balanced(self):
        """Test that smart tier falls back to balanced when smart not configured."""
        defaults = {"models": {"balanced": "balanced-model", "fast": "fast-model"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_model_from_tier(
                defaults, tier="smart", user_kwargs={}, plugin_name="test"
            )

        assert result == "balanced-model"
        # Should warn about fallback
        assert len(w) == 1
        assert "balanced" in str(w[0].message).lower()

    def test_smart_fallback_to_fast(self):
        """Test that smart tier falls back to fast when smart and balanced not configured."""
        defaults = {"models": {"fast": "fast-model"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_model_from_tier(
                defaults, tier="smart", user_kwargs={}, plugin_name="test"
            )

        assert result == "fast-model"
        assert len(w) == 1
        assert "fast" in str(w[0].message).lower()

    def test_balanced_fallback_to_fast(self):
        """Test that balanced tier falls back to fast when balanced not configured."""
        defaults = {"models": {"smart": "smart-model", "fast": "fast-model"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_model_from_tier(
                defaults, tier="balanced", user_kwargs={}, plugin_name="test"
            )

        assert result == "fast-model"
        assert len(w) == 1
        assert "fast" in str(w[0].message).lower()

    def test_balanced_fallback_to_smart(self):
        """Test that balanced tier falls back to smart when balanced and fast not configured."""
        defaults = {"models": {"smart": "smart-model"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_model_from_tier(
                defaults, tier="balanced", user_kwargs={}, plugin_name="test"
            )

        assert result == "smart-model"
        assert len(w) == 1
        assert "smart" in str(w[0].message).lower()
        assert "costlier" in str(w[0].message).lower()

    def test_fast_fallback_to_balanced(self):
        """Test that fast tier falls back to balanced when fast not configured."""
        defaults = {"models": {"smart": "smart-model", "balanced": "balanced-model"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_model_from_tier(
                defaults, tier="fast", user_kwargs={}, plugin_name="test"
            )

        assert result == "balanced-model"
        assert len(w) == 1
        assert "balanced" in str(w[0].message).lower()

    def test_fast_fallback_to_smart(self):
        """Test that fast tier falls back to smart when fast and balanced not configured."""
        defaults = {"models": {"smart": "smart-model"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_model_from_tier(
                defaults, tier="fast", user_kwargs={}, plugin_name="test"
            )

        assert result == "smart-model"
        assert len(w) == 1
        assert "smart" in str(w[0].message).lower()
        assert "costlier" in str(w[0].message).lower()

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_no_models_defined(self):
        """Test behavior when no models are defined."""
        defaults = {"temperature": 0.7}

        result = _resolve_model_from_tier(
            defaults, tier="smart", user_kwargs={}, plugin_name="test"
        )

        assert result is None

    def test_single_model_default(self):
        """Test behavior with single model (no tiers)."""
        defaults = {"model": "default-model"}

        result = _resolve_model_from_tier(
            defaults, tier="smart", user_kwargs={}, plugin_name="test"
        )

        assert result == "default-model"

    def test_user_models_override_defaults(self):
        """Test that user-provided models override defaults."""
        defaults = {"models": {"smart": "default-smart", "fast": "default-fast"}}
        user_kwargs = {"models": {"smart": "user-smart"}}

        result = _resolve_model_from_tier(
            defaults, tier="smart", user_kwargs=user_kwargs, plugin_name="test"
        )

        assert result == "user-smart"

    def test_partial_user_override(self):
        """Test that partial user override merges with defaults."""
        defaults = {
            "models": {
                "smart": "default-smart",
                "fast": "default-fast",
                "balanced": "default-balanced",
            }
        }
        user_kwargs = {"models": {"smart": "user-smart"}}  # Only override smart

        # Smart should use user override
        result = _resolve_model_from_tier(
            defaults, tier="smart", user_kwargs=user_kwargs, plugin_name="test"
        )
        assert result == "user-smart"

        # Fast should use default
        result = _resolve_model_from_tier(
            defaults, tier="fast", user_kwargs=user_kwargs, plugin_name="test"
        )
        assert result == "default-fast"

    def test_empty_models_dict(self):
        """Test behavior with empty models dict."""
        defaults = {"models": {}}

        result = _resolve_model_from_tier(
            defaults, tier="smart", user_kwargs={}, plugin_name="test"
        )

        assert result is None

    def test_no_warning_when_tier_not_specified_and_fallback(self):
        """Test that no warning is issued when tier is None but fallback happens."""
        defaults = {"models": {"balanced": "balanced-model"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_model_from_tier(
                defaults, tier=None, user_kwargs={}, plugin_name="test"
            )

        # tier=None, so no warning even though fallback happened
        assert result == "balanced-model"
        assert len(w) == 0
