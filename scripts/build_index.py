"""Step 3: embed the evidence corpus (MiniLM) and build the FAISS index. Use a GPU if you have one."""
from traceverify import config, data
from traceverify.retrieval import Retriever, load_embedder


def main():
    lookup = data.load_evidence_lookup()
    r = Retriever.build(lookup, load_embedder())
    r.save(config.RAG_DIR)
    print(f"indexed {r.index.ntotal} passages -> {config.RAG_DIR}")


if __name__ == "__main__":
    main()
