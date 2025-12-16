from fitz.core.config.loader import _load_yaml
from fitz.core.config.loader import DEFAULT_CONFIG_PATH


def test_default_yaml_is_meta_config():
    data = _load_yaml(DEFAULT_CONFIG_PATH)

    assert "default_preset" in data
    assert "presets" in data
    assert isinstance(data["presets"], dict)
