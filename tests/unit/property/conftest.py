# tests/unit/property/conftest.py
"""
Hypothesis configuration and profiles for property-based tests.

Profiles:
    - ci: 500 examples, for CI/nightly runs
    - dev: 100 examples, for local development (default)
    - quick: 20 examples, for smoke tests
"""

import pytest
from hypothesis import Phase, Verbosity, settings

# Register Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=500,
    verbosity=Verbosity.normal,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink],
    deadline=None,  # No time limit in CI
)

settings.register_profile(
    "dev",
    max_examples=100,
    verbosity=Verbosity.normal,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink],
    deadline=1000,  # 1 second per example
)

settings.register_profile(
    "quick",
    max_examples=20,
    verbosity=Verbosity.quiet,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    deadline=500,  # 500ms per example
)

# Default to dev profile
settings.load_profile("dev")


def pytest_configure(config):
    """Configure Hypothesis profile based on pytest options."""
    # Check for --hypothesis-profile option
    profile = getattr(config.option, "hypothesis_profile", None)
    if profile:
        settings.load_profile(profile)


@pytest.fixture(scope="session")
def hypothesis_profile():
    """Return the current Hypothesis profile name."""
    return settings._current_profile
