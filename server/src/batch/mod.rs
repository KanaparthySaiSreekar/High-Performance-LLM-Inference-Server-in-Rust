use crate::error::{Result, ServerError};
use crate::inference::{GenerationParams, InferenceEngine};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Semaphore};
use uuid::Uuid;

/// Request submitted to the batch queue
#[derive(Debug)]
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub params: GenerationParams,
    pub response_tx: oneshot::Sender<Result<String>>,
}

impl InferenceRequest {
    pub fn new(
        prompt: String,
        params: GenerationParams,
    ) -> (Self, oneshot::Receiver<Result<String>>) {
        let (tx, rx) = oneshot::channel();
        let request = Self {
            id: Uuid::new_v4().to_string(),
            prompt,
            params,
            response_tx: tx,
        };
        (request, rx)
    }
}

/// Batch processor manages a queue of inference requests
pub struct BatchProcessor {
    pub(crate) request_tx: mpsc::Sender<InferenceRequest>,
    pub(crate) capacity_semaphore: Arc<Semaphore>,
}

impl BatchProcessor {
    pub fn new(
        engine: Arc<InferenceEngine>,
        queue_capacity: usize,
        max_batch_size: usize,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::channel::<InferenceRequest>(queue_capacity);
        let capacity_semaphore = Arc::new(Semaphore::new(queue_capacity));

        // Spawn worker tasks
        for worker_id in 0..max_batch_size {
            let mut rx = request_rx.clone();
            let engine = Arc::clone(&engine);
            let semaphore = Arc::clone(&capacity_semaphore);

            tokio::spawn(async move {
                tracing::info!("Worker {} started", worker_id);

                while let Some(request) = rx.recv().await {
                    tracing::debug!(
                        "Worker {} processing request {}",
                        worker_id,
                        request.id
                    );

                    let result = engine.generate(&request.prompt, request.params).await;

                    // Send response back
                    let _ = request.response_tx.send(result);

                    // Release capacity
                    semaphore.add_permits(1);
                }

                tracing::info!("Worker {} stopped", worker_id);
            });
        }

        // Drop the original receiver so workers own all clones
        drop(request_rx);

        Self {
            request_tx,
            capacity_semaphore,
        }
    }

    /// Submit a request for processing
    pub async fn submit(
        &self,
        prompt: String,
        params: GenerationParams,
    ) -> Result<oneshot::Receiver<Result<String>>> {
        // Acquire permit (blocks if queue is full)
        let permit = self
            .capacity_semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| ServerError::QueueFull)?;

        // Create request
        let (request, response_rx) = InferenceRequest::new(prompt, params);

        // Submit to queue
        self.request_tx
            .send(request)
            .await
            .map_err(|_| ServerError::Internal("Failed to submit request".to_string()))?;

        // Permit will be released when worker completes the request
        forget(permit);

        Ok(response_rx)
    }

    /// Get the current queue size
    pub fn queue_size(&self) -> usize {
        self.capacity_semaphore.available_permits()
    }
}

// Helper to forget the permit (we'll release it manually in workers)
fn forget<T>(_t: T) {
    // Intentionally does nothing - we manage permits manually
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_request_creation() {
        let (request, _rx) = InferenceRequest::new(
            "test prompt".to_string(),
            GenerationParams::default(),
        );

        assert_eq!(request.prompt, "test prompt");
        assert!(!request.id.is_empty());
    }
}
