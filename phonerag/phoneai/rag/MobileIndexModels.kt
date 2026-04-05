package com.example.myapplication.rag

data class MobileManifest(
    val version: Int,
    val createdAtUtc: String,
    val numChunks: Int,
    val embeddingDim: Int,
    val embeddingDtype: String
)

data class MobileChunk(
    val text: String,
    val metadataJson: String,
    val docId: String,
    val chunkId: Int
)

data class RetrievalHit(
    val chunk: MobileChunk,
    val score: Float
)
