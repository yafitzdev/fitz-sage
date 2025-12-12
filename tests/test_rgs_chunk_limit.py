from rag.generation.rgs import RGS, RGSConfig


def test_rgs_respects_max_chunks():
    cfg = RGSConfig(max_chunks=2)
    rgs = RGS(cfg)

    chunks = [
        {"id": "1", "text": "A", "metadata": {}},
        {"id": "2", "text": "B", "metadata": {}},
        {"id": "3", "text": "C", "metadata": {}},
    ]

    prompt = rgs.build_prompt("Q?", chunks)

    # Only two chunks should appear
    assert "[S1]" in prompt.user
    assert "[S2]" in prompt.user
    assert "[S3]" not in prompt.user
