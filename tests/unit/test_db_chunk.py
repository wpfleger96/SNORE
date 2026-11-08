"""Unit tests for the chunked-ID-binding helper."""

import pytest

from snore.utils import db_chunk
from snore.utils.db_chunk import iter_id_chunks


class TestIterIdChunks:
    def test_empty_input_yields_nothing(self) -> None:
        assert list(iter_id_chunks([])) == []

    def test_within_cap_yields_single_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db_chunk, "ID_CHUNK_SIZE", 500)
        chunks = list(iter_id_chunks(list(range(500))))
        assert len(chunks) == 1
        assert chunks[0] == tuple(range(500))

    def test_exact_multiple_of_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(db_chunk, "ID_CHUNK_SIZE", 2)
        chunks = list(iter_id_chunks([1, 2, 3, 4]))
        assert chunks == [(1, 2), (3, 4)]

    def test_remainder_yields_short_final_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db_chunk, "ID_CHUNK_SIZE", 2)
        chunks = list(iter_id_chunks([1, 2, 3, 4, 5]))
        assert chunks == [(1, 2), (3, 4), (5,)]

    def test_chunks_are_tuples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(db_chunk, "ID_CHUNK_SIZE", 2)
        for chunk in iter_id_chunks([1, 2, 3]):
            assert isinstance(chunk, tuple)

    def test_reads_module_global_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Proves monkeypatching the module global (not a bound default arg) takes effect.
        monkeypatch.setattr(db_chunk, "ID_CHUNK_SIZE", 3)
        assert list(iter_id_chunks([1, 2, 3, 4])) == [(1, 2, 3), (4,)]

    def test_explicit_size_yields_smaller_chunks(self) -> None:
        assert list(iter_id_chunks([1, 2, 3, 4, 5], 2)) == [(1, 2), (3, 4), (5,)]
