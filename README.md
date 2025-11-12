# High-Performance LLM Inference Server in Rust

A production-ready, high-performance Large Language Model (LLM) inference server built in Rust. Delivers ultra-fast, scalable AI services with an OpenAI-compatible API.

> **Note**: This is a proof-of-concept implementation showcasing production-grade architecture for LLM serving. See [DEVELOPMENT.md](DEVELOPMENT.md) for build instructions and known issues.

## 🚀 Features

- **High Performance**: Zero-cost abstractions and true parallelism for low-latency inference
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Concurrent Request Handling**: Async request batching and queue management with Tokio
- **Multiple Model Architectures**: Support for Llama, Mistral, Phi, and more
- **Flexible Model Loading**: Load from local files or HuggingFace Hub
- **Memory Efficient**: Memory-mapped file loading and optional quantization
- **Observability**: Prometheus metrics and structured logging
- **Production Ready**: Error handling, health checks, and graceful shutdown

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HTTP API Server (Axum)                   │
├─────────────────────────────────────────────────────────────┤
│  /v1/completions  │  /v1/chat/completions  │  /v1/models   │
├─────────────────────────────────────────────────────────────┤
│                   Request Queue & Batching                  │
├─────────────────────────────────────────────────────────────┤
│                   Worker Pool (Tokio)                       │
├─────────────────────────────────────────────────────────────┤
│              Inference Engine (Candle)                      │
├─────────────────────────────────────────────────────────────┤
│  Model Loading  │  Tokenization  │  Text Generation         │
├─────────────────────────────────────────────────────────────┤
│            Backend (CPU / CUDA / Metal)                     │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Model Loading Module** (`src/model/`)
   - Safetensors support with memory-mapped I/O
   - HuggingFace Hub integration
   - Multiple architecture support

2. **Inference Engine** (`src/inference/`)
   - Text generation with advanced sampling (temperature, top-p, top-k)
   - KV-cache management for efficiency
   - Repetition penalty and stop sequences

3. **HTTP API Server** (`src/api/`)
   - OpenAI-compatible endpoints
   - Request validation and error handling
   - CORS and compression middleware

4. **Batch Processor** (`src/batch/`)
   - Concurrent request queue
   - Worker pool management
   - Backpressure handling

5. **Metrics** (`src/metrics/`)
   - Prometheus metrics
   - Request latency histograms
   - Token generation counters

## 📦 Installation

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- CUDA 11.8+ (optional, for GPU acceleration)
- A compatible LLM model (e.g., Llama 2, Mistral)

### Build from Source

```bash
git clone https://github.com/KanaparthySaiSreekar/High-Performance-LLM-Inference-Server-in-Rust
cd High-Performance-LLM-Inference-Server-in-Rust
cargo build --release
```

### With CUDA Support

```bash
cargo build --release --features cuda
```

## 🎯 Quick Start

### 1. Download a Model

```bash
# Using HuggingFace CLI
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir ./models/llama-2-7b

# Or use any compatible model in safetensors format
```

### 2. Configure the Server

```bash
cp config.example.toml config.toml
# Edit config.toml with your model path
```

### 3. Start the Server

```bash
# Using configuration file
./target/release/llm-server --config config.toml

# Or using CLI arguments
./target/release/llm-server \
  --model-path ./models/llama-2-7b \
  --model-arch llama \
  --port 8080 \
  --device cpu
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Text completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Rust?"}
    ],
    "max_tokens": 150
  }'
```

## 📖 API Documentation

### OpenAI-Compatible Endpoints

#### POST /v1/completions

Generate text completions.

**Request:**
```json
{
  "prompt": "string",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stop": ["string"],
  "stream": false
}
```

**Response:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "llama",
  "choices": [{
    "text": "string",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

#### POST /v1/chat/completions

Generate chat completions.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  }
}
```

#### GET /v1/models

List available models.

#### GET /health

Health check endpoint.

#### GET /metrics

Prometheus metrics endpoint.

## ⚙️ Configuration

### Command Line Arguments

```bash
Options:
      --host <HOST>                    Host to bind to [env: LLM_HOST=] [default: 0.0.0.0]
  -p, --port <PORT>                    Port to bind to [env: LLM_PORT=] [default: 8080]
  -m, --model-path <MODEL_PATH>        Path to model directory [env: LLM_MODEL_PATH=]
      --model-arch <MODEL_ARCH>        Model architecture [env: LLM_MODEL_ARCH=] [default: llama]
      --tokenizer-path <PATH>          Path to tokenizer [env: LLM_TOKENIZER_PATH=]
      --max-batch-size <SIZE>          Maximum batch size [env: LLM_MAX_BATCH_SIZE=] [default: 16]
      --max-seq-len <LENGTH>           Maximum sequence length [env: LLM_MAX_SEQ_LEN=] [default: 2048]
      --device <DEVICE>                Device (cpu, cuda, metal) [env: LLM_DEVICE=] [default: cpu]
      --quantization <QUANT>           Quantization (none, q4_0, q8_0) [env: LLM_QUANTIZATION=] [default: none]
      --workers <N>                    Worker threads [env: LLM_WORKERS=] [default: 4]
      --queue-capacity <N>             Queue capacity [env: LLM_QUEUE_CAPACITY=] [default: 100]
      --log-level <LEVEL>              Log level [env: LLM_LOG_LEVEL=] [default: info]
  -c, --config <CONFIG>                Configuration file [env: LLM_CONFIG=]
  -h, --help                           Print help
  -V, --version                        Print version
```

### Environment Variables

All CLI arguments can be set via environment variables with the `LLM_` prefix:

```bash
export LLM_MODEL_PATH="./models/llama-2-7b"
export LLM_PORT=8080
export LLM_DEVICE=cuda
./target/release/llm-server
```

### Configuration File

See `config.example.toml` for a complete configuration example.

## 🔧 Performance Tuning

### Memory Optimization

1. **Quantization**: Use INT8 or INT4 quantization to reduce memory footprint
   ```bash
   --quantization q8_0  # 8-bit quantization
   ```

2. **Batch Size**: Adjust based on available memory
   ```bash
   --max-batch-size 8  # Smaller batches for limited memory
   ```

### Throughput Optimization

1. **Workers**: Increase worker threads for concurrent processing
   ```bash
   --workers 8  # More workers for higher throughput
   ```

2. **Queue Capacity**: Larger queues handle traffic spikes
   ```bash
   --queue-capacity 200
   ```

### GPU Acceleration

```bash
# CUDA (NVIDIA)
--device cuda

# Metal (Apple Silicon)
--device metal
```

## 📊 Monitoring

### Prometheus Metrics

Available at `/metrics`:

- `llm_requests_total` - Total number of requests
- `llm_requests_success_total` - Successful requests
- `llm_requests_failed_total` - Failed requests
- `llm_request_duration_seconds` - Request latency histogram
- `llm_tokens_generated_total` - Total tokens generated
- `llm_queue_size` - Current queue size
- `llm_active_requests` - Active requests being processed

### Logging

Structured logging with configurable levels:

```bash
--log-level debug  # trace, debug, info, warn, error
```

## 🧪 Development

### Run Tests

```bash
cargo test
```

### Run Benchmarks

```bash
cargo bench
```

### Code Coverage

```bash
cargo tarpaulin --out Html
```

## 🛣️ Roadmap

### Short-term
- [x] Core inference engine with Candle
- [x] OpenAI-compatible API
- [x] Request batching and queue management
- [x] Prometheus metrics
- [ ] Streaming responses (SSE)
- [ ] Multi-GPU support

### Medium-term
- [ ] Advanced quantization (GPTQ, AWQ)
- [ ] Model parallelism for large models
- [ ] Distributed serving across nodes
- [ ] Flash Attention optimization
- [ ] Speculative decoding

### Long-term
- [ ] Multimodal support (text + image)
- [ ] On-device deployment
- [ ] Fine-tuning integration
- [ ] RAG (Retrieval-Augmented Generation)
- [ ] Content safety guardrails

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Candle](https://github.com/huggingface/candle) - ML framework by HuggingFace
- [Tokio](https://tokio.rs) - Async runtime
- [Axum](https://github.com/tokio-rs/axum) - Web framework
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - Inspiration for Rust LLM serving

## 📚 References

- [Efficient LLM Inference](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.ai/)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)

## 💬 Support

- GitHub Issues: [Report bugs or request features](https://github.com/KanaparthySaiSreekar/High-Performance-LLM-Inference-Server-in-Rust/issues)
- Discussions: [Ask questions and share ideas](https://github.com/KanaparthySaiSreekar/High-Performance-LLM-Inference-Server-in-Rust/discussions)

---

Built with ❤️ and Rust 🦀
