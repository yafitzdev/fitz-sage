from fitz.core.config.loader import load_config
from fitz.core.config.schema import FitzConfig


def test_default_config_loads_and_validates_base_schema():
    """
    This test verifies the CURRENT architectural contract:

    - default.yaml can be loaded
    - it validates against FitzConfig
    - only keys declared in FitzConfig are accepted

    If this test fails, config loading is broken.
    If this test passes, but the system is incomplete, that is expected.
    """

    cfg = load_config()

    assert isinstance(cfg, FitzConfig)
    assert hasattr(cfg, "llm")
    assert cfg.llm.plugin_name is not None
