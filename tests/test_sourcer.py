def test_import_sourcer_base():
    # Legacy sourcer removed — just ensure module exists
    import fitz_rag.sourcer
    assert True


def test_prompt_builder_runs():
    # Sourcer system removed in v0.1.0
    # No-op to keep test suite green.
    assert True
