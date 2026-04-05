package com.example.myapplication.rag

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

class MobileIndexLoader(private val indexDir: File) {
    data class LoadedIndex(
        val manifest: MobileManifest,
        val chunks: List<MobileChunk>,
        val embeddings: FloatArray
    )

    fun load(): LoadedIndex {
        val manifestFile = File(indexDir, "manifest.json")
        val chunksFile = File(indexDir, "chunks.json")
        val embeddingsFile = File(indexDir, "embeddings.f32")

        require(manifestFile.exists()) { "Missing manifest.json in ${indexDir.absolutePath}" }
        require(chunksFile.exists()) { "Missing chunks.json in ${indexDir.absolutePath}" }
        require(embeddingsFile.exists()) { "Missing embeddings.f32 in ${indexDir.absolutePath}" }

        val manifest = parseManifest(manifestFile.readText(Charsets.UTF_8))
        val chunks = parseChunks(chunksFile.readText(Charsets.UTF_8))
        val embeddings = parseFloat32File(embeddingsFile)

        require(chunks.size == manifest.numChunks) {
            "Chunk mismatch: manifest=${manifest.numChunks}, chunks=${chunks.size}"
        }
        require(embeddings.size == manifest.numChunks * manifest.embeddingDim) {
            "Embedding shape mismatch: values=${embeddings.size}, expected=${manifest.numChunks * manifest.embeddingDim}"
        }

        return LoadedIndex(manifest = manifest, chunks = chunks, embeddings = embeddings)
    }

    private fun parseManifest(json: String): MobileManifest {
        val obj = JSONObject(json)
        return MobileManifest(
            version = obj.getInt("version"),
            createdAtUtc = obj.optString("created_at_utc", ""),
            numChunks = obj.getInt("num_chunks"),
            embeddingDim = obj.getInt("embedding_dim"),
            embeddingDtype = obj.getString("embedding_dtype")
        )
    }

    private fun parseChunks(json: String): List<MobileChunk> {
        val arr = JSONArray(json)
        val out = ArrayList<MobileChunk>(arr.length())
        for (i in 0 until arr.length()) {
            val item = arr.getJSONObject(i)
            out += MobileChunk(
                text = item.getString("text"),
                metadataJson = item.optJSONObject("metadata")?.toString() ?: "{}",
                docId = item.getString("doc_id"),
                chunkId = item.getInt("chunk_id")
            )
        }
        return out
    }

    private fun parseFloat32File(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size % 4 == 0) { "embeddings.f32 byte size must be multiple of 4." }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val count = bytes.size / 4
        val out = FloatArray(count)
        for (i in 0 until count) {
            out[i] = bb.float
        }
        return out
    }
}
