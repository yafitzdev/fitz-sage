import pytest
from fitz_ingest.ingester.validation import IngestionValidator, ChunkValidationConfig, IngestionValidationError

def test_ingestion_validator_required_metadata_keys():
    cfg = ChunkValidationConfig(required_metadata_keys=("file",))
    validator = IngestionValidator(cfg)

    bad_chunk = [{"text": "hello", "metadata": {}}]

    with pytest.raises(IngestionValidationError):
        validator.validate_chunks(bad_chunk, "dummy.txt")

    good_chunk = [{"text": "hello", "metadata": {"file": "x"}}]

    # should NOT raise
    validator.validate_chunks(good_chunk, "dummy.txt")
