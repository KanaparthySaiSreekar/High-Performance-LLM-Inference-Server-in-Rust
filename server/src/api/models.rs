use serde::{Deserialize, Serialize};

/// OpenAI-compatible completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// The model to use (ignored for now, as we only support one model at a time)
    #[serde(default)]
    pub model: String,

    /// The prompt to generate from
    pub prompt: String,

    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Sampling temperature
    #[serde(default)]
    pub temperature: Option<f64>,

    /// Top-p sampling
    #[serde(default)]
    pub top_p: Option<f64>,

    /// Top-k sampling
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: usize,

    /// Stream the response
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,

    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: Option<f64>,

    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: Option<f64>,

    /// User identifier
    #[serde(default)]
    pub user: Option<String>,
}

fn default_n() -> usize {
    1
}

/// OpenAI-compatible completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// Chat completion request (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// The model to use
    #[serde(default)]
    pub model: String,

    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,

    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Sampling temperature
    #[serde(default)]
    pub temperature: Option<f64>,

    /// Top-p sampling
    #[serde(default)]
    pub top_p: Option<f64>,

    /// Number of chat completion choices to generate
    #[serde(default = "default_n")]
    pub n: usize,

    /// Stream the response
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,

    /// User identifier
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Chat completion response (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// List of models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<Model>,
}

impl ChatCompletionRequest {
    /// Convert chat messages to a single prompt string
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();

        for message in &self.messages {
            match message.role.as_str() {
                "system" => {
                    prompt.push_str(&format!("System: {}\n\n", message.content));
                }
                "user" => {
                    prompt.push_str(&format!("User: {}\n\n", message.content));
                }
                "assistant" => {
                    prompt.push_str(&format!("Assistant: {}\n\n", message.content));
                }
                _ => {
                    prompt.push_str(&format!("{}: {}\n\n", message.role, message.content));
                }
            }
        }

        prompt.push_str("Assistant: ");
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_to_prompt() {
        let request = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are a helpful assistant.".to_string(),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Hello!".to_string(),
                    name: None,
                },
            ],
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: 1,
            stream: false,
            stop: None,
            user: None,
        };

        let prompt = request.to_prompt();
        assert!(prompt.contains("System: You are a helpful assistant."));
        assert!(prompt.contains("User: Hello!"));
        assert!(prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_completion_request_defaults() {
        let json = r#"{"prompt": "test"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.n, 1);
        assert!(!req.stream);
    }
}
