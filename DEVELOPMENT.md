# Development Guide

## Project Status

This is a **proof-of-concept** implementation of a high-performance LLM inference server in Rust. The architecture is production-ready, but there are some dependency version conflicts that need resolution for compilation.

## Known Issues

### Candle Version Compatibility

The current configuration uses `candle-core = "0.7"` which has some compatibility issues with the `rand` crate and `half` crate in certain environments. This is a known issue in the Candle ecosystem.

**Workarounds:**

1. **Use Candle 0.6.x** (more stable):
   ```toml
   candle-core = "0.6"
   candle-nn = "0.6"
   candle-transformers = "0.6"
   ```

2. **Use Candle 0.9.x** (latest):
   ```toml
   candle-core = "0.9"
   candle-nn = "0.9"
   candle-transformers = "0.9"
   ```
   Note: May require code changes for API differences

3. **Use a Candle fork** or wait for upstream fixes

### Rand Crate Conflict

The `rand` crate has multiple versions in the dependency tree (0.8.x and 0.9.x). We've pinned to `rand = "=0.8.5"` to match Candle's expectations, but transitive dependencies may still pull in 0.9.x.

**Solution:** Use cargo's `[patch]` section to force a specific version across all dependencies.

## Current Implementation Status

### ✅ Completed Components

1. **Core Architecture**
   - [x] Model loading module with safetensors support
   - [x] Inference engine with advanced sampling strategies
   - [x] HTTP API server with Axum
   - [x] Request batching and queue management
   - [x] Prometheus metrics integration
   - [x] Configuration management (CLI + config files)

2. **API Endpoints**
   - [x] `/v1/completions` - Text completion (OpenAI-compatible)
   - [x] `/v1/chat/completions` - Chat completion (OpenAI-compatible)
   - [x] `/v1/models` - List available models
   - [x] `/health` - Health check
   - [x] `/metrics` - Prometheus metrics

3. **Performance Features**
   - [x] Async request handling with Tokio
   - [x] Worker pool for concurrent processing
   - [x] Queue backpressure handling
   - [x] Memory-mapped file loading for models

4. **Observability**
   - [x] Structured logging with tracing
   - [x] Prometheus metrics (requests, latency, tokens)
   - [x] Health checks

### 🚧 In Progress

1. **Streaming Responses**
   - Infrastructure in place but not fully implemented
   - Need to integrate with actual model generation loop

2. **Model Quantization**
   - Configuration options present
   - Implementation requires Candle quantization features

### 📋 TODO

1. **Resolve Dependency Conflicts**
   - Pin exact versions of all ML crates
   - Or upgrade to latest Candle version

2. **Multi-GPU Support**
   - Extend device selection to support multiple GPUs
   - Implement model sharding

3. **Advanced Optimizations**
   - Flash Attention integration
   - Speculative decoding
   - KV-cache optimization

## Development Workflow

### Building the Project

#### Option 1: Demo Mode (No Actual Model)

The current `main.rs` includes a demo mode that works without actual model files:

```bash
cargo build --release
./target/release/llm-server --model-path dummy --port 8080
```

This will start a server that echoes requests without actual inference.

#### Option 2: Production Mode (With Models)

1. Download a compatible model:
   ```bash
   huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
     --local-dir ./models/llama-2-7b
   ```

2. Uncomment the production initialization code in `main.rs`

3. Build and run:
   ```bash
   cargo build --release
   ./target/release/llm-server \
     --model-path ./models/llama-2-7b \
     --model-arch llama
   ```

### Testing

```bash
# Run unit tests
cargo test

# Run integration tests (requires model)
cargo test --test integration_tests

# Run benchmarks
cargo bench
```

### Code Structure

```
server/src/
├── main.rs              # Entry point and server initialization
├── config.rs            # Configuration management
├── error.rs             # Error types and handling
├── api/
│   ├── mod.rs           # Router and middleware setup
│   ├── handlers.rs      # HTTP request handlers
│   └── models.rs        # Request/response types
├── model/
│   ├── mod.rs           # Model abstraction
│   ├── loader.rs        # Model file loading
│   └── tokenizer.rs     # Tokenization
├── inference/
│   ├── mod.rs           # Inference engine
│   ├── sampler.rs       # Sampling strategies
│   └── generator.rs     # Text generation
├── batch/
│   └── mod.rs           # Request batching
└── metrics/
    └── mod.rs           # Prometheus metrics
```

## Performance Tuning

### Memory Optimization

1. **Model Quantization**: Reduce model size from FP32 to INT8/INT4
2. **Memory Mapping**: Use `memmap2` for zero-copy model loading
3. **KV Cache Management**: Efficiently reuse cached key-value pairs

### Throughput Optimization

1. **Batch Size**: Increase for higher throughput (at cost of latency)
2. **Worker Threads**: Match to CPU core count
3. **Queue Capacity**: Size based on expected load

### Latency Optimization

1. **Model Selection**: Smaller models for lower latency
2. **Batch Size**: Decrease for lower latency
3. **GPU Acceleration**: Use CUDA/Metal for faster inference

## Deployment

### Docker

```bash
# Build image
docker build -t llm-inference-server .

# Run container
docker run -p 8080:8080 \
  -v ./models:/models:ro \
  -e LLM_MODEL_PATH=/models/llama-2-7b \
  llm-inference-server
```

### Docker Compose

```bash
# Start server
docker-compose up -d

# With monitoring stack
docker-compose --profile monitoring up -d
```

### Kubernetes

See `k8s/` directory for Kubernetes manifests (TODO).

## Contributing

### Code Style

- Use `rustfmt` for formatting: `cargo fmt`
- Use `clippy` for linting: `cargo clippy`
- Write tests for new features
- Update documentation

### Pull Request Process

1. Create a feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation
5. Submit PR with clear description

## Troubleshooting

### Compilation Errors

**Problem**: `candle-core` fails to compile with rand/half trait errors

**Solution**:
- Try Candle 0.6.x or wait for 0.7.x fixes
- Or use a different ML framework (e.g., tch-rs for PyTorch bindings)

**Problem**: Out of memory during model loading

**Solution**:
- Use quantization (`--quantization q8_0`)
- Use a smaller model
- Increase system RAM or use GPU memory

### Runtime Errors

**Problem**: "Model path does not exist"

**Solution**:
- Verify model path is correct
- Ensure model files are in safetensors format
- Download model if using HuggingFace ID

**Problem**: Slow inference

**Solution**:
- Enable GPU acceleration (`--device cuda`)
- Reduce sequence length
- Use quantized model

## Resources

- [Candle Documentation](https://github.com/huggingface/candle)
- [Axum Guide](https://docs.rs/axum/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [LLM Optimization Guide](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

## License

MIT - See LICENSE file for details.
