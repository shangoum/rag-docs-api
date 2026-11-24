import pytest

from app.services.chunking import chunk_pages


def test_short_pages_give_one_chunk_each():
    chunks = chunk_pages(["First page.", "", "Third   page\n text."], size=100, overlap=10)
    assert [(c.page, c.text) for c in chunks] == [(1, "First page."), (3, "Third page text.")]
    assert [c.position for c in chunks] == [0, 1]


def test_long_page_is_split_on_word_boundaries_with_overlap():
    words = [f"word{i}" for i in range(200)]
    chunks = chunk_pages([" ".join(words)], size=120, overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 120
        # every chunk starts and ends with a whole word
        assert all(token in words for token in chunk.text.split())
    # consecutive chunks share some words (overlap)
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert set(first.text.split()) & set(second.text.split())
    # nothing is lost
    covered = {token for c in chunks for token in c.text.split()}
    assert covered == set(words)


def test_chunks_never_cross_pages():
    chunks = chunk_pages(["alpha " * 50, "beta " * 50], size=80, overlap=20)
    assert all("alpha" not in c.text for c in chunks if c.page == 2)
    assert all("beta" not in c.text for c in chunks if c.page == 1)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(["text"], size=size, overlap=overlap)
