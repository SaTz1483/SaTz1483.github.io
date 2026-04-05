package com.example.myapplication.rag

import android.content.Context
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class DebugMode {
    RETRIEVAL,
    CHAT
}

class RetrievalRepository(indexDir: File) {
    private val loadedIndex = MobileIndexLoader(indexDir).load()
    private val retriever = CosineRetriever(
        chunks = loadedIndex.chunks,
        embeddings = loadedIndex.embeddings,
        embeddingDim = loadedIndex.manifest.embeddingDim
    )

    val embeddingDim: Int
        get() = loadedIndex.manifest.embeddingDim

    fun searchByEmbedding(
        queryEmbedding: FloatArray,
        topK: Int,
        minSimilarity: Float
    ): List<RetrievalHit> {
        return retriever.search(
            queryEmbedding = queryEmbedding,
            topK = topK,
            minSimilarity = minSimilarity
        )
    }
}

data class DebugRetrievalUiState(
    val loading: Boolean = false,
    val error: String? = null,
    val embeddingDim: Int = 0,
    val hits: List<RetrievalHit> = emptyList()
)

data class ChatUiState(
    val loading: Boolean = false,
    val error: String? = null,
    val answer: String = "",
    val sources: List<RetrievalHit> = emptyList(),
    val modelPath: String? = null,
    val modelReady: Boolean = false
)

object PromptBuilder {
    fun build(question: String, hits: List<RetrievalHit>): String {
        val context = hits.joinToString("\n\n") { it.chunk.text }
        return "Use only the context below to answer the question. If the answer is not in the context, say you do not know.\n\nContext:\n$context\n\nQuestion: $question"
    }
}

interface LocalGenerator {
    suspend fun generate(prompt: String): String
    fun destroy() {}
}

object FakeGenerator : LocalGenerator {
    override suspend fun generate(prompt: String): String {
        return "Stubbed local generation response. Prompt length=${prompt.length} chars."
    }
}

class LlamaGenerator(
    context: Context,
    private val systemPrompt: String = "You are a concise document assistant. Use only provided context."
) : LocalGenerator {
    private val appContext = context.applicationContext
    private val engine: InferenceEngine = AiChat.getInferenceEngine(context)
    private var loadedModelPath: String? = null
    private var systemPromptSet = false

    suspend fun importModelFromUri(uri: Uri): String = withContext(Dispatchers.IO) {
        val modelsDir = File(appContext.filesDir, "models").also {
            if (!it.exists()) it.mkdirs()
        }

        val targetFile = File(modelsDir, "selected_model.gguf")
        appContext.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(targetFile).use { output ->
                input.copyTo(output)
            }
        } ?: throw IllegalStateException("Unable to open itselected model file")

        loadedModelPath = null
        systemPromptSet = false
        targetFile.absolutePath
    }

    suspend fun loadModel(modelPath: String) = withContext(Dispatchers.IO) {
        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            throw IllegalStateException("Model file not found: $modelPath")
        }
        if (!modelFile.isFile) {
            throw IllegalStateException("Model path is not a file: $modelPath")
        }
        if (!modelFile.canRead()) {
            throw IllegalStateException("Model file is not readable: $modelPath")
        }

        if (loadedModelPath != modelPath) {
            engine.loadModel(modelPath)
            loadedModelPath = modelPath
            systemPromptSet = false
        }
        if (!systemPromptSet && systemPrompt.isNotBlank()) {
            engine.setSystemPrompt(systemPrompt)
            systemPromptSet = true
        }
    }

    private suspend fun ensureLoaded() = withContext(Dispatchers.IO) {
        val modelPath = loadedModelPath
            ?: throw IllegalStateException("No model loaded. Pick a GGUF model first.")
        loadModel(modelPath)
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        ensureLoaded()
        engine.sendUserPrompt(prompt, predictLength = 256)
            .take(256)
            .toList()
            .joinToString(separator = "")
            .trim()
    }

    override fun destroy() {
        engine.destroy()
    }
}

class DebugRetrievalViewModel(
    private val repository: RetrievalRepository,
    private val generator: LocalGenerator = FakeGenerator
) : ViewModel() {
    private val _uiState = MutableStateFlow(
        DebugRetrievalUiState(embeddingDim = repository.embeddingDim)
    )
    val uiState: StateFlow<DebugRetrievalUiState> = _uiState.asStateFlow()
    private val _chatUiState = MutableStateFlow(ChatUiState())
    val chatUiState: StateFlow<ChatUiState> = _chatUiState.asStateFlow()

    fun importAndLoadModel(uri: Uri) {
        _chatUiState.value = _chatUiState.value.copy(
            loading = true,
            error = null,
            answer = ""
        )

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val llamaGenerator = generator as? LlamaGenerator
                    ?: throw IllegalStateException("Local generator is not configured for GGUF chat")
                val modelPath = llamaGenerator.importModelFromUri(uri)
                llamaGenerator.loadModel(modelPath)

                _chatUiState.value = _chatUiState.value.copy(
                    loading = false,
                    error = null,
                    modelPath = modelPath,
                    modelReady = true
                )
            } catch (e: Exception) {
                _chatUiState.value = _chatUiState.value.copy(
                    loading = false,
                    modelReady = false,
                    error = "${e::class.java.simpleName}: ${e.message ?: "no message"}"
                )
            }
        }
    }

    fun askDirect(question: String) {
        _chatUiState.value = _chatUiState.value.copy(
            loading = true,
            error = null,
            answer = "",
            sources = emptyList()
        )

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val answer = generator.generate(question)
                _chatUiState.value = _chatUiState.value.copy(
                    loading = false,
                    answer = answer,
                    error = null
                )
            } catch (e: Exception) {
                _chatUiState.value = _chatUiState.value.copy(
                    loading = false,
                    answer = "",
                    error = "${e::class.java.simpleName}: ${e.message ?: "no message"}"
                )
            }
        }
    }

    fun runSearch(
        rawEmbedding: String,
        topK: Int,
        minSimilarity: Float
    ) {
        _uiState.value = _uiState.value.copy(loading = true, error = null)
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val vector = parseEmbedding(rawEmbedding)
                if (vector.size != repository.embeddingDim) {
                    throw IllegalArgumentException(
                        "Embedding dim mismatch: got=${vector.size}, expected=${repository.embeddingDim}"
                    )
                }
                val results = repository.searchByEmbedding(
                    queryEmbedding = vector,
                    topK = topK,
                    minSimilarity = minSimilarity
                )
                _uiState.value = _uiState.value.copy(
                    loading = false,
                    hits = results,
                    error = null
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loading = false,
                    hits = emptyList(),
                    error = e.message ?: "Unknown error"
                )
            }
        }
    }

    fun askQuestion(
        question: String,
        rawEmbedding: String,
        topK: Int,
        minSimilarity: Float
    ) {
        _chatUiState.value = _chatUiState.value.copy(loading = true, error = null, answer = "")
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val vector = parseEmbedding(rawEmbedding)
                if (vector.size != repository.embeddingDim) {
                    throw IllegalArgumentException(
                        "Embedding dim mismatch: got=${vector.size}, expected=${repository.embeddingDim}"
                    )
                }

                val hits = repository.searchByEmbedding(
                    queryEmbedding = vector,
                    topK = topK,
                    minSimilarity = minSimilarity
                )

                if (hits.isEmpty()) {
                    _chatUiState.value = _chatUiState.value.copy(
                        loading = false,
                        answer = "I do not know based on the indexed documents.",
                        sources = emptyList(),
                        error = null
                    )
                    return@launch
                }

                val prompt = PromptBuilder.build(question, hits)
                val answer = generator.generate(prompt)

                _chatUiState.value = _chatUiState.value.copy(
                    loading = false,
                    answer = answer,
                    sources = hits,
                    error = null
                )
            } catch (e: Exception) {
                _chatUiState.value = _chatUiState.value.copy(
                    loading = false,
                    answer = "",
                    sources = emptyList(),
                    error = e.message ?: "Unknown error"
                )
            }
        }
    }

    private fun parseEmbedding(raw: String): FloatArray {
        val values = raw
            .split(",", " ", "\n", "\t")
            .mapNotNull { token ->
                val trimmed = token.trim()
                if (trimmed.isEmpty()) null else trimmed.toFloatOrNull()
            }
        if (values.isEmpty()) {
            throw IllegalArgumentException("No numeric values found in embedding input.")
        }
        return values.toFloatArray()
    }

    override fun onCleared() {
        generator.destroy()
        super.onCleared()
    }
}

@Composable
fun DebugRetrievalScreen(
    viewModel: DebugRetrievalViewModel
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val chatState by viewModel.chatUiState.collectAsStateWithLifecycle()
    var mode by remember { mutableStateOf(DebugMode.RETRIEVAL) }
    var rawEmbedding by remember { mutableStateOf("") }
    var question by remember { mutableStateOf("") }
    var topK by remember { mutableIntStateOf(3) }
    var minSimilarity by remember { mutableStateOf("0.25") }
    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { viewModel.importAndLoadModel(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "RAG Debug Console",
            style = MaterialTheme.typography.titleLarge
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = { mode = DebugMode.RETRIEVAL },
                modifier = Modifier.weight(1f)
            ) { Text("Retrieval") }
            Button(
                onClick = { mode = DebugMode.CHAT },
                modifier = Modifier.weight(1f)
            ) { Text("Chat") }
        }
        Text(
            text = "Expected embedding dim: ${state.embeddingDim}",
            style = MaterialTheme.typography.bodyMedium
        )

        if (mode == DebugMode.CHAT) {
            Button(
                onClick = { modelPicker.launch(arrayOf("*/*")) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (chatState.modelReady) "Change GGUF Model" else "Pick GGUF Model")
            }
            Text(
                text = chatState.modelPath ?: "No model selected",
                style = MaterialTheme.typography.bodySmall
            )
            OutlinedTextField(
                value = question,
                onValueChange = { question = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Question") }
            )
            Button(
                onClick = { viewModel.askDirect(question) },
                enabled = chatState.modelReady && question.isNotBlank() && !chatState.loading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Ask")
            }
        } else {
            OutlinedTextField(
                value = rawEmbedding,
                onValueChange = { rawEmbedding = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(160.dp),
                label = { Text("Query embedding values (comma/space separated)") },
                minLines = 4,
                maxLines = 6
            )
            OutlinedTextField(
                value = topK.toString(),
                onValueChange = { input -> topK = input.toIntOrNull() ?: topK },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Top K") }
            )
            OutlinedTextField(
                value = minSimilarity,
                onValueChange = { minSimilarity = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Min similarity") }
            )

            Button(
                onClick = {
                    val threshold = minSimilarity.toFloatOrNull() ?: 0.0f
                    viewModel.runSearch(rawEmbedding, topK, threshold)
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Run Retrieval")
            }
        }

        if (mode == DebugMode.RETRIEVAL) {
            state.error?.let { err ->
                Text(
                    text = err,
                    color = MaterialTheme.colorScheme.error
                )
            }
            if (state.loading) {
                Text("Running retrieval...")
            }
        } else {
            chatState.error?.let { err ->
                Text(
                    text = err,
                    color = MaterialTheme.colorScheme.error
                )
            }
            if (chatState.loading) {
                Text("Generating answer...")
            }
            if (chatState.answer.isNotBlank()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text("Answer", style = MaterialTheme.typography.titleMedium)
                        Text(chatState.answer)
                    }
                }
            }
        }

        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            val hits = if (mode == DebugMode.RETRIEVAL) state.hits else chatState.sources
            items(hits) { hit ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text("Score: %.3f".format(hit.score))
                        Text(
                            text = "Doc: ${hit.chunk.docId} | Chunk: ${hit.chunk.chunkId}",
                            style = MaterialTheme.typography.bodySmall
                        )
                        Text(
                            text = hit.chunk.text,
                            maxLines = 4,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
    }
}
