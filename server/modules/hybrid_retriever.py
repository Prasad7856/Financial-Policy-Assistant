from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import json
import re

BM25_FILE = "data/bm25_chunks.json"


class HybridRetriever:

    def __init__(self, pinecone_index, embedding_model):
        self.index = pinecone_index
        self.embed_model = embedding_model

        self.documents = []
        self.doc_lookup = {}

        with open(BM25_FILE, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                document = Document(
                    page_content=item["text"],
                    metadata={
                        "chunk_id": item.get("chunk_id"),
                        "document_id": item.get("document_id"),
                        "filename": item.get("filename"),
                        "source": item.get("source", ""),
                        "page": item.get("page", 0)
                    }
                )

                self.documents.append(document)

                self.doc_lookup[
                    item.get("chunk_id")
                ] = document

        if not self.documents:
            raise ValueError(
                "No documents found in bm25_chunks.json. "
                "Upload PDFs again to build BM25 index."
            )

        tokenized_docs = [
            re.findall(r"\w+", doc.page_content.lower())
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

    @staticmethod
    def normalize_scores(score_dict):
        """
        Min-Max Normalization
        """

        if not score_dict:
            return {}

        values = list(score_dict.values())

        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            return {
                k: 1.0
                for k in score_dict
            }

        return {
            k: (v - min_score) /
            (max_score - min_score)
            for k, v in score_dict.items()
        }

    def retrieve(self, query, top_k=20):

        # ===================================
        # Dense Retrieval (Pinecone)
        # ===================================

        query_embedding = self.embed_model.embed_query(
            query
        )

        dense_results = self.index.query(
            vector=query_embedding,
            top_k=20,
            include_metadata=True
        )

        dense_scores = {}
        dense_docs = {}

        print("\n========== DENSE RESULTS ==========")

        for match in dense_results["matches"]:

            metadata = match["metadata"]

            chunk_id = metadata["chunk_id"]

            doc = Document(
                page_content=metadata["text"],
                metadata=metadata
            )

            dense_docs[chunk_id] = doc

            dense_scores[chunk_id] = float(
                match["score"]
            )

            print(
                f"Chunk={chunk_id} | "
                f"Score={match['score']:.4f}"
            )

        # ===================================
        # BM25 Retrieval
        # ===================================

        query_tokens = re.findall(
            r"\w+",
            query.lower()
        )

        bm25_scores_array = self.bm25.get_scores(
            query_tokens
        )

        bm25_scores = {}

        print("\n========== BM25 RESULTS ==========")

        for idx, score in enumerate(
            bm25_scores_array
        ):

            if score <= 0:
                continue

            doc = self.documents[idx]

            chunk_id = doc.metadata[
                "chunk_id"
            ]

            bm25_scores[chunk_id] = float(
                score
            )

        sorted_bm25 = sorted(
            bm25_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        for chunk_id, score in sorted_bm25:

            print(
                f"Chunk={chunk_id} | "
                f"Score={score:.4f}"
            )

        bm25_scores = dict(sorted_bm25)

        # ===================================
        # Normalize Scores
        # ===================================

        dense_scores = self.normalize_scores(
            dense_scores
        )

        bm25_scores = self.normalize_scores(
            bm25_scores
        )

        # ===================================
        # Hybrid Score Fusion
        # ===================================

        hybrid_results = {}

        all_chunk_ids = set(
            dense_scores.keys()
        ).union(
            bm25_scores.keys()
        )

        for chunk_id in all_chunk_ids:

            dense_score = dense_scores.get(
                chunk_id,
                0.0
            )

            bm25_score = bm25_scores.get(
                chunk_id,
                0.0
            )

            hybrid_score = (
                0.7 * dense_score +
                0.3 * bm25_score
            )

            if chunk_id in dense_docs:

                doc = dense_docs[chunk_id]

            else:

                doc = self.doc_lookup[
                    chunk_id
                ]

            doc.metadata[
                "dense_score"
            ] = round(
                dense_score,
                4
            )

            doc.metadata[
                "bm25_score"
            ] = round(
                bm25_score,
                4
            )

            doc.metadata[
                "hybrid_score"
            ] = round(
                hybrid_score,
                4
            )

            hybrid_results[
                chunk_id
            ] = doc

        # ===================================
        # Final Ranking
        # ===================================

        ranked_docs = sorted(
            hybrid_results.values(),
            key=lambda d:
            d.metadata[
                "hybrid_score"
            ],
            reverse=True
        )

        final_docs = ranked_docs[:top_k]

        print(
            "\n========== FINAL HYBRID RESULTS =========="
        )

        for i, doc in enumerate(
            final_docs,
            start=1
        ):

            print(
                f"\nRank #{i}"
                f"\nChunk ID     : {doc.metadata.get('chunk_id')}"
                f"\nHybrid Score : {doc.metadata.get('hybrid_score')}"
                f"\nDense Score  : {doc.metadata.get('dense_score')}"
                f"\nBM25 Score   : {doc.metadata.get('bm25_score')}"
                f"\nSource       : {doc.metadata.get('filename')}"
                f"\nPage         : {doc.metadata.get('page')}"
            )

            print(
                doc.page_content[:250]
            )

        return final_docs