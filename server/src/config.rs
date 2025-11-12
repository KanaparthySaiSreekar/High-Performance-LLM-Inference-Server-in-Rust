use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "High-Performance LLM Inference Server", long_about = None)]
pub struct CliArgs {
    /// Host to bind to
    #[arg(long, env = "LLM_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Port to bind to
    #[arg(short, long, env = "LLM_PORT", default_value = "8080")]
    pub port: u16,

    /// Path to model directory or Hugging Face model ID
    #[arg(short, long, env = "LLM_MODEL_PATH")]
    pub model_path: String,

    /// Model architecture (llama, mistral, phi, etc.)
    #[arg(long, env = "LLM_MODEL_ARCH", default_value = "llama")]
    pub model_arch: String,

    /// Path to tokenizer configuration
    #[arg(long, env = "LLM_TOKENIZER_PATH")]
    pub tokenizer_path: Option<String>,

    /// Maximum batch size for concurrent requests
    #[arg(long, env = "LLM_MAX_BATCH_SIZE", default_value = "16")]
    pub max_batch_size: usize,

    /// Maximum sequence length
    #[arg(long, env = "LLM_MAX_SEQ_LEN", default_value = "2048")]
    pub max_seq_len: usize,

    /// Device to use (cpu, cuda, metal)
    #[arg(long, env = "LLM_DEVICE", default_value = "cpu")]
    pub device: String,

    /// Use quantization (none, q4_0, q4_1, q8_0)
    #[arg(long, env = "LLM_QUANTIZATION", default_value = "none")]
    pub quantization: String,

    /// Number of worker threads for inference
    #[arg(long, env = "LLM_WORKERS", default_value = "4")]
    pub workers: usize,

    /// Request queue capacity
    #[arg(long, env = "LLM_QUEUE_CAPACITY", default_value = "100")]
    pub queue_capacity: usize,

    /// Enable metrics endpoint
    #[arg(long, env = "LLM_METRICS", default_value = "true")]
    pub enable_metrics: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "LLM_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Configuration file path
    #[arg(short, long, env = "LLM_CONFIG")]
    pub config: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model: ModelConfig,
    pub inference: InferenceConfig,
    pub performance: PerformanceConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub architecture: String,
    pub tokenizer_path: Option<String>,
    pub device: String,
    pub quantization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_seq_len: usize,
    pub default_temperature: f64,
    pub default_top_p: f64,
    pub default_top_k: usize,
    pub default_max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_batch_size: usize,
    pub workers: usize,
    pub queue_capacity: usize,
    pub enable_kv_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub enable_metrics: bool,
    pub json_output: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            model: ModelConfig::default(),
            inference: InferenceConfig::default(),
            performance: PerformanceConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            architecture: "llama".to_string(),
            tokenizer_path: None,
            device: "cpu".to_string(),
            quantization: "none".to_string(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            default_temperature: 0.7,
            default_top_p: 0.9,
            default_top_k: 50,
            default_max_tokens: 256,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            workers: 4,
            queue_capacity: 100,
            enable_kv_cache: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            enable_metrics: true,
            json_output: false,
        }
    }
}

impl From<CliArgs> for ServerConfig {
    fn from(args: CliArgs) -> Self {
        Self {
            host: args.host,
            port: args.port,
            model: ModelConfig {
                path: args.model_path,
                architecture: args.model_arch,
                tokenizer_path: args.tokenizer_path,
                device: args.device,
                quantization: args.quantization,
            },
            inference: InferenceConfig {
                max_seq_len: args.max_seq_len,
                ..Default::default()
            },
            performance: PerformanceConfig {
                max_batch_size: args.max_batch_size,
                workers: args.workers,
                queue_capacity: args.queue_capacity,
                ..Default::default()
            },
            logging: LoggingConfig {
                level: args.log_level,
                enable_metrics: args.enable_metrics,
                ..Default::default()
            },
        }
    }
}

impl ServerConfig {
    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ServerConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn merge_with_cli(mut self, args: CliArgs) -> Self {
        // CLI args override config file
        self.host = args.host;
        self.port = args.port;
        self.model.path = args.model_path;
        self.model.architecture = args.model_arch;
        if args.tokenizer_path.is_some() {
            self.model.tokenizer_path = args.tokenizer_path;
        }
        self.model.device = args.device;
        self.model.quantization = args.quantization;
        self.inference.max_seq_len = args.max_seq_len;
        self.performance.max_batch_size = args.max_batch_size;
        self.performance.workers = args.workers;
        self.performance.queue_capacity = args.queue_capacity;
        self.logging.level = args.log_level;
        self.logging.enable_metrics = args.enable_metrics;
        self
    }
}
