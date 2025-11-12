use crate::api::models::*;
use crate::api::AppState;
use crate::error::{Result, ServerError};
use crate::inference::GenerationParams;
use crate::metrics::MetricsTracker;
use axum::{extract::State, Json};
use uuid::Uuid;

/// Handle /v1/completions endpoint
pub async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>> {
    let tracker = MetricsTracker::start();

    tracing::info!("Received completion request for prompt: {:?}", request.prompt);

    // Validate request
    if request.prompt.is_empty() {
        return Err(ServerError::InvalidRequest("Prompt cannot be empty".to_string()));
    }

    if request.n != 1 {
        return Err(ServerError::InvalidRequest(
            "Currently only n=1 is supported".to_string(),
        ));
    }

    // Build generation parameters
    let params = GenerationParams {
        max_tokens: request.max_tokens.unwrap_or(256),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k.unwrap_or(50),
        repetition_penalty: 1.0 + request.presence_penalty.unwrap_or(0.0),
        stop: request.stop.unwrap_or_default(),
        stream: request.stream,
    };

    // Submit to batch processor
    let response_rx = state
        .batch_processor
        .submit(request.prompt.clone(), params)
        .await?;

    // Wait for response
    let generated_text = response_rx
        .await
        .map_err(|_| ServerError::Internal("Response channel closed".to_string()))??;

    // Estimate token counts (rough approximation)
    let prompt_tokens = request.prompt.split_whitespace().count();
    let completion_tokens = generated_text.split_whitespace().count();

    tracker.success(completion_tokens);

    // Build response
    let response = CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: state.config.model.architecture.clone(),
        choices: vec![CompletionChoice {
            text: generated_text,
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response))
}

/// Handle /v1/chat/completions endpoint
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>> {
    let tracker = MetricsTracker::start();

    tracing::info!("Received chat completion request with {} messages", request.messages.len());

    // Validate request
    if request.messages.is_empty() {
        return Err(ServerError::InvalidRequest("Messages cannot be empty".to_string()));
    }

    if request.n != 1 {
        return Err(ServerError::InvalidRequest(
            "Currently only n=1 is supported".to_string(),
        ));
    }

    // Convert chat messages to prompt
    let prompt = request.to_prompt();

    // Build generation parameters
    let params = GenerationParams {
        max_tokens: request.max_tokens.unwrap_or(256),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: 50,
        repetition_penalty: 1.0,
        stop: request.stop.unwrap_or_default(),
        stream: request.stream,
    };

    // Submit to batch processor
    let response_rx = state
        .batch_processor
        .submit(prompt.clone(), params)
        .await?;

    // Wait for response
    let generated_text = response_rx
        .await
        .map_err(|_| ServerError::Internal("Response channel closed".to_string()))??;

    // Estimate token counts
    let prompt_tokens = prompt.split_whitespace().count();
    let completion_tokens = generated_text.split_whitespace().count();

    tracker.success(completion_tokens);

    // Build response
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: state.config.model.architecture.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: generated_text,
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response))
}

/// Handle /v1/models endpoint
pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![Model {
            id: state.config.model.architecture.clone(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "local".to_string(),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_params_from_request() {
        let request = CompletionRequest {
            model: "test".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(40),
            n: 1,
            stream: false,
            stop: None,
            presence_penalty: Some(0.5),
            frequency_penalty: None,
            user: None,
        };

        let params = GenerationParams {
            max_tokens: request.max_tokens.unwrap_or(256),
            temperature: request.temperature.unwrap_or(0.7),
            top_p: request.top_p.unwrap_or(0.9),
            top_k: request.top_k.unwrap_or(50),
            repetition_penalty: 1.0 + request.presence_penalty.unwrap_or(0.0),
            stop: request.stop.unwrap_or_default(),
            stream: request.stream,
        };

        assert_eq!(params.max_tokens, 100);
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.repetition_penalty, 1.5);
    }
}
