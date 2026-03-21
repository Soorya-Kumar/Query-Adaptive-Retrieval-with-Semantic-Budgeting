from storage.store_postgres import fetch_tags_for_document

def test_fetch_tags_for_document():
    doc_id = "doc2"
    tags = fetch_tags_for_document(doc_id)
    print(f"Tags for document {doc_id}: {tags}")