package com.example.myapplication.rag


import kotlin.math.sqrt

class CosineRetriever(
    private val chunks: List<MobileChunk>,
    private val embeddings: FloatArray,
    private val embeddingDim: Int
) {
    init {
        require(chunks.isNotEmpty()) { "No chunks loaded." }
        require(embeddingDim > 0) { "embeddingDim must be > 0." }
        require(embeddings.size == chunks.size * embeddingDim) {
            "Embeddings size mismatch."
        }
    }

    fun search(queryEmbedding: FloatArray, topK: Int, minSimilarity: Float): List<RetrievalHit> {
        require(queryEmbedding.size == embeddingDim) {
            "Query embedding dim mismatch: got=${queryEmbedding.size}, expected=$embeddingDim"
        }
        require(topK > 0) { "topK must be > 0." }

        val queryNorm = l2Norm(queryEmbedding)
        if (queryNorm == 0f) return emptyList()

        val scored = ArrayList<RetrievalHit>(chunks.size)
        for (i in chunks.indices) {
            val rowOffset = i * embeddingDim
            val score = cosineAt(rowOffset, queryEmbedding, queryNorm)
            if (score >= minSimilarity) {
                scored += RetrievalHit(chunks[i], score)
            }
        }

        return scored
            .sortedByDescending { it.score }
            .take(topK)
    }

    private fun cosineAt(rowOffset: Int, query: FloatArray, queryNorm: Float): Float {
        var dot = 0.0f
        var embNormSq = 0.0f
        for (j in 0 until embeddingDim) {
            val e = embeddings[rowOffset + j]
            dot += e * query[j]
            embNormSq += e * e
        }
        val embNorm = sqrt(embNormSq)
        if (embNorm == 0f) return 0f
        return dot / (queryNorm * embNorm)
    }

    private fun l2Norm(v: FloatArray): Float {
        var s = 0.0f
        for (x in v) {
            s += x * x
        }
        return sqrt(s)
    }
}
