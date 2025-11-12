use lazy_static::lazy_static;
use prometheus::{
    Counter, Encoder, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry, TextEncoder,
};
use std::sync::Arc;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Request counters
    pub static ref REQUESTS_TOTAL: IntCounter = IntCounter::new(
        "llm_requests_total",
        "Total number of inference requests"
    )
    .expect("metric creation failed");

    pub static ref REQUESTS_SUCCESS: IntCounter = IntCounter::new(
        "llm_requests_success_total",
        "Total number of successful requests"
    )
    .expect("metric creation failed");

    pub static ref REQUESTS_FAILED: IntCounter = IntCounter::new(
        "llm_requests_failed_total",
        "Total number of failed requests"
    )
    .expect("metric creation failed");

    // Latency histogram
    pub static ref REQUEST_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "llm_request_duration_seconds",
            "Request duration in seconds"
        )
        .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0])
    )
    .expect("metric creation failed");

    // Token metrics
    pub static ref TOKENS_GENERATED: Counter = Counter::with_opts(
        Opts::new(
            "llm_tokens_generated_total",
            "Total number of tokens generated"
        )
    )
    .expect("metric creation failed");

    // Queue metrics
    pub static ref QUEUE_SIZE: IntGauge = IntGauge::new(
        "llm_queue_size",
        "Current number of requests in queue"
    )
    .expect("metric creation failed");

    pub static ref ACTIVE_REQUESTS: IntGauge = IntGauge::new(
        "llm_active_requests",
        "Number of requests currently being processed"
    )
    .expect("metric creation failed");
}

pub fn register_metrics() {
    REGISTRY
        .register(Box::new(REQUESTS_TOTAL.clone()))
        .expect("failed to register metric");
    REGISTRY
        .register(Box::new(REQUESTS_SUCCESS.clone()))
        .expect("failed to register metric");
    REGISTRY
        .register(Box::new(REQUESTS_FAILED.clone()))
        .expect("failed to register metric");
    REGISTRY
        .register(Box::new(REQUEST_DURATION.clone()))
        .expect("failed to register metric");
    REGISTRY
        .register(Box::new(TOKENS_GENERATED.clone()))
        .expect("failed to register metric");
    REGISTRY
        .register(Box::new(QUEUE_SIZE.clone()))
        .expect("failed to register metric");
    REGISTRY
        .register(Box::new(ACTIVE_REQUESTS.clone()))
        .expect("failed to register metric");
}

pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Metrics middleware for tracking request metrics
pub struct MetricsTracker {
    start_time: std::time::Instant,
}

impl MetricsTracker {
    pub fn start() -> Self {
        REQUESTS_TOTAL.inc();
        ACTIVE_REQUESTS.inc();
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    pub fn success(self, _token_count: usize) {
        REQUESTS_SUCCESS.inc();
        ACTIVE_REQUESTS.dec();
        let duration = self.start_time.elapsed().as_secs_f64();
        REQUEST_DURATION.observe(duration);
    }

    pub fn failure(self) {
        REQUESTS_FAILED.inc();
        ACTIVE_REQUESTS.dec();
        let duration = self.start_time.elapsed().as_secs_f64();
        REQUEST_DURATION.observe(duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registration() {
        register_metrics();
        let metrics = gather_metrics();
        assert!(metrics.contains("llm_requests_total"));
    }
}
