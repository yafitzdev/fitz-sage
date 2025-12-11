from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_metadata_truncation():
    cfg = RGSConfig()
    rgs = RGS(config=cfg)

    chunks = [
        {
            "id": "1",
            "text": "hello",
            "metadata": {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,  # fourth entry → should trigger "..."
            },
        }
    ]

    prompt = rgs.build_prompt("q", chunks)

    # Expect only first 3 items + ellipsis marker
    assert "(metadata:" in prompt.user
    assert "a=" in prompt.user
    assert "b=" in prompt.user
    assert "c=" in prompt.user
    assert "d=" not in prompt.user
    assert "..." in prompt.user
