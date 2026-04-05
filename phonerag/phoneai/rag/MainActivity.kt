package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.UUID
import kotlin.math.ln
import kotlin.math.sqrt

data class SourceChunk(
    val id: String,
    val title: String,
    val text: String
)

data class RetrievalHit(
    val chunk: SourceChunk,
    val score: Double
)

class MainActivity : AppCompatActivity() {

    private lateinit var statusTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton

    private lateinit var engine: InferenceEngine
    private var generationJob: Job? = null

    private var isModelReady = false
    private val messages = mutableListOf<Message>()
    private val lastAssistantMsg = StringBuilder()
    private val messageAdapter = MessageAdapter(messages)

    private val knowledgeBase = listOf(
        SourceChunk(
            id = "S1",
            title = "Meeting Note",
            text = "Rahul is in Conference Room A at 5 PM in Chennai office."
        ),
        SourceChunk(
            id = "S2",
            title = "Schedule",
            text = "The demo starts at 5:30 PM and Ananya is presenting."
        ),
        SourceChunk(
            id = "S3",
            title = "Location Info",
            text = "Conference Room A is on the second floor near the reception desk."
        )
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        onBackPressedDispatcher.addCallback {
            Log.w(TAG, "Ignoring back press")
        }

        statusTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)

        messagesRv.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        messagesRv.adapter = messageAdapter

        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
        }

        userActionFab.setOnClickListener {
            if (isModelReady) {
                handleUserInput()
            } else {
                getContent.launch(arrayOf("*/*"))
            }
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { handleSelectedModel(it) }
    }

    private fun handleSelectedModel(uri: Uri) {
        userActionFab.isEnabled = false
        userInputEt.hint = "Parsing GGUF..."
        statusTv.text = "Importing model..."

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d("ABI", android.os.Build.SUPPORTED_ABIS.joinToString())

                val metadata = contentResolver.openInputStream(uri)?.use {
                    GgufMetadataReader.create().readStructuredMetadata(it)
                } ?: throw IllegalStateException("Failed to parse GGUF metadata")

                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                val modelFile = contentResolver.openInputStream(uri)?.use { input ->
                    ensureModelFile(modelName, input)
                } ?: throw IllegalStateException("Failed to import selected model")

                loadModel(modelName, modelFile)

                withContext(Dispatchers.Main) {
                    isModelReady = true
                    userInputEt.isEnabled = true
                    userInputEt.hint = "Ask a question"
                    userActionFab.isEnabled = true
                    userActionFab.setImageResource(R.drawable.outline_send_24)
                    statusTv.text = "Model loaded: $modelName"
                    Toast.makeText(this@MainActivity, "Model loaded successfully", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                withContext(Dispatchers.Main) {
                    isModelReady = false
                    userActionFab.isEnabled = true
                    userInputEt.isEnabled = false
                    userInputEt.hint = "Pick a GGUF model first"
                    statusTv.text = "${e::class.java.simpleName}: ${e.message ?: "no message"}"
                    Toast.makeText(
                        this@MainActivity,
                        "${e::class.java.simpleName}: ${e.message ?: "no message"}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private suspend fun ensureModelFile(modelName: String, input: InputStream): File =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { file ->
                if (!file.exists()) {
                    FileOutputStream(file).use { output -> input.copyTo(output) }
                }
            }
        }

    private suspend fun loadModel(modelName: String, modelFile: File) =
        withContext(Dispatchers.IO) {
            Log.i(TAG, "Loading model $modelName from ${modelFile.path}")
            withContext(Dispatchers.Main) {
                userInputEt.hint = "Loading model..."
            }
            engine.loadModel(modelFile.path)
            engine.setSystemPrompt(
                "Answer only from the provided context. If the answer is missing, say you do not know."
            )
        }

    private fun tokenize(text: String): List<String> {
        return text.lowercase().split(Regex("\\W+")).filter { it.length > 2 }
    }

    private fun cosineSimilarity(a: DoubleArray, b: DoubleArray): Double {
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        if (normA == 0.0 || normB == 0.0) return 0.0
        return dot / (sqrt(normA) * sqrt(normB))
    }

    private fun retrieveSources(question: String, topK: Int = 3): List<RetrievalHit> {
        val documents = knowledgeBase.map { tokenize(it.text) }
        val queryTokens = tokenize(question)
        if (queryTokens.isEmpty()) return emptyList()

        val vocabulary = (documents.flatten() + queryTokens).distinct()
        val vocabIndex = vocabulary.withIndex().associate { it.value to it.index }
        val docCount = documents.size.toDouble()
        val documentFrequencies = IntArray(vocabulary.size)

        documents.forEach { tokens ->
            tokens.toSet().forEach { token ->
                val idx = vocabIndex[token] ?: return@forEach
                documentFrequencies[idx]++
            }
        }

        fun tfidfVector(tokens: List<String>): DoubleArray {
            val counts = tokens.groupingBy { it }.eachCount()
            val totalTerms = tokens.size.toDouble().coerceAtLeast(1.0)
            val vector = DoubleArray(vocabulary.size)

            counts.forEach { (token, count) ->
                val idx = vocabIndex[token] ?: return@forEach
                val tf = count / totalTerms
                val df = documentFrequencies[idx].toDouble()
                val idf = ln((docCount + 1.0) / (df + 1.0)) + 1.0
                vector[idx] = tf * idf
            }
            return vector
        }

        val queryVector = tfidfVector(queryTokens)

        return knowledgeBase.mapIndexed { index, chunk ->
            val docVector = tfidfVector(documents[index])
            RetrievalHit(chunk, cosineSimilarity(queryVector, docVector))
        }
            .filter { it.score > 0.01 }
            .sortedByDescending { it.score }
            .take(topK)
    }

    private fun buildRagPrompt(question: String, hits: List<RetrievalHit>): String {
        val contextBlock = hits.joinToString("\n") {
            "${it.chunk.id}: ${it.chunk.text}"
        }

        return """
            Context:
            $contextBlock

            Question: $question

            Answer in one short sentence.
            Then write:
            Sources:
        """.trimIndent()
    }

    private fun handleUserInput() {
        val userMsg = userInputEt.text.toString().trim()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, "Input message is empty", Toast.LENGTH_SHORT).show()
            return
        }

        userInputEt.text = null
        userInputEt.isEnabled = false
        userActionFab.isEnabled = false

        messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
        lastAssistantMsg.clear()
        messages.add(Message(UUID.randomUUID().toString(), "", false))
        messageAdapter.notifyDataSetChanged()
        messagesRv.scrollToPosition(messages.lastIndex)

        generationJob = lifecycleScope.launch(Dispatchers.Default) {
            try {
                val hits = retrieveSources(userMsg)
                Log.d("RAG_HITS", hits.joinToString("\n") {
                    "${it.chunk.id} | ${it.chunk.title} | score=${it.score} | text=${it.chunk.text}"
                })

                val finalPrompt = if (hits.isEmpty()) {
                    "Question: $userMsg\n\nAnswer: I do not know based on the provided sources."
                } else {
                    buildRagPrompt(userMsg, hits)
                }

                Log.d("RAG_PROMPT", finalPrompt)

                engine.sendUserPrompt(finalPrompt)
                    .onCompletion {
                        withContext(Dispatchers.Main) {
                            userInputEt.isEnabled = true
                            userActionFab.isEnabled = true
                            statusTv.text = if (hits.isNotEmpty()) {
                                "Sources used:\n" + hits.joinToString("\n") {
                                    "${it.chunk.id}: ${it.chunk.title}"
                                }
                            } else {
                                "No matching sources found"
                            }
                        }
                    }
                    .collect { token ->
                        withContext(Dispatchers.Main) {
                            val lastIndex = messages.lastIndex
                            val current = messages[lastIndex]
                            messages[lastIndex] = current.copy(
                                content = lastAssistantMsg.append(token).toString()
                            )
                            messageAdapter.notifyItemChanged(lastIndex)
                            messagesRv.scrollToPosition(messages.lastIndex)
                        }
                    }
            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                withContext(Dispatchers.Main) {
                    userInputEt.isEnabled = true
                    userActionFab.isEnabled = true
                    statusTv.text = "${e::class.java.simpleName}: ${e.message ?: "no message"}"
                    Toast.makeText(
                        this@MainActivity,
                        "${e::class.java.simpleName}: ${e.message ?: "no message"}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private fun ensureModelsDirectory(): File =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) it.delete()
            if (!it.exists()) it.mkdir()
        }

    override fun onStop() {
        generationJob?.cancel()
        super.onStop()
    }

    override fun onDestroy() {
        if (::engine.isInitialized) {
            engine.destroy()
        }
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"
    }
}

fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size -> "$name-$size" } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { arch ->
            basic.uuid?.let { uuid -> "$arch-$uuid" } ?: "$arch-${System.currentTimeMillis()}"
        }
    }
    else -> {
        "model-${System.currentTimeMillis()}"
    }
}
