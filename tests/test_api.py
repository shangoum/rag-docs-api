from helpers import upload

from app.services.openai_http import LLMError


async def test_health(client):
    assert (await client.get("/health")).json() == {"status": "ok"}


async def test_upload_pdf_indexes_pages_and_chunks(client, handbook_pdf):
    response = await upload(client, "handbook.pdf", handbook_pdf)
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "handbook.pdf"
    assert body["pages"] == 3
    assert body["chunks"] >= 3

    documents = (await client.get("/documents")).json()
    assert [d["id"] for d in documents] == [body["id"]]


async def test_ask_returns_answer_with_page_citation(client, handbook_pdf):
    await upload(client, "handbook.pdf", handbook_pdf)

    response = await client.post("/ask", json={"question": "What is the notice period?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    top = body["sources"][0]
    assert top["filename"] == "handbook.pdf"
    assert top["page"] == 2
    assert "notice period" in top["excerpt"]
    assert "[1]" in body["answer"]


async def test_unrelated_question_is_not_answered(client, handbook_pdf):
    await upload(client, "handbook.pdf", handbook_pdf)
    response = await client.post("/ask", json={"question": "What is the capital of Peru?"})
    body = response.json()
    assert body["found"] is False
    assert body["sources"] == []
    assert "could not find" in body["answer"]


async def test_model_not_called_without_relevant_passages(make_client, handbook_pdf):
    class SpyChat:
        calls = 0

        async def answer(self, system_prompt: str, user_prompt: str) -> str:
            SpyChat.calls += 1
            return "should not be used"

    async for client in make_client(chat=SpyChat()):
        await upload(client, "handbook.pdf", handbook_pdf)
        await client.post("/ask", json={"question": "Who won the football match?"})
    assert SpyChat.calls == 0


async def test_ask_can_be_restricted_to_documents(client, handbook_pdf):
    other = await upload(client, "recipes.txt", b"Pancakes need flour, milk and eggs.")
    await upload(client, "handbook.pdf", handbook_pdf)

    response = await client.post(
        "/ask",
        json={"question": "What is the notice period?", "document_ids": [other.json()["id"]]},
    )
    assert response.json()["found"] is False


async def test_delete_document_removes_its_chunks(client, handbook_pdf):
    created = (await upload(client, "handbook.pdf", handbook_pdf)).json()

    assert (await client.delete(f"/documents/{created['id']}")).status_code == 204
    assert (await client.get("/documents")).json() == []
    body = (await client.post("/ask", json={"question": "What is the notice period?"})).json()
    assert body["found"] is False
    assert (await client.delete(f"/documents/{created['id']}")).status_code == 404


async def test_text_and_markdown_files_are_accepted(client):
    response = await upload(client, "notes.md", b"# Notes\nThe server restarts every Sunday.")
    assert response.status_code == 201
    assert response.json()["pages"] == 1


async def test_unsupported_file_type(client):
    response = await upload(client, "photo.png", b"\x89PNG...")
    assert response.status_code == 415


async def test_pdf_without_text_is_rejected(client):
    response = await upload(client, "scan.pdf", b"not really a pdf")
    assert response.status_code == 422


async def test_empty_and_too_large_files(make_client):
    async for client in make_client(max_upload_mb=1):
        assert (await upload(client, "empty.txt", b"")).status_code == 400
        assert (await upload(client, "big.txt", b"a" * (1024 * 1024 + 1))).status_code == 413


async def test_client_path_is_stripped_from_filename(client):
    response = await upload(client, "../../etc/notes.txt", b"Some useful content here.")
    assert response.json()["filename"] == "notes.txt"


async def test_provider_error_returns_502(make_client, handbook_pdf):
    class DownChat:
        async def answer(self, system_prompt: str, user_prompt: str) -> str:
            raise LLMError("Model provider returned HTTP 503")

    async for client in make_client(chat=DownChat()):
        await upload(client, "handbook.pdf", handbook_pdf)
        response = await client.post("/ask", json={"question": "What is the notice period?"})
        assert response.status_code == 502
        assert response.json()["detail"] == "Model provider returned HTTP 503"


async def test_question_validation(client):
    assert (await client.post("/ask", json={"question": "?"})).status_code == 422
    assert (await client.post("/ask", json={"question": "Valid?", "top_k": 50})).status_code == 422
