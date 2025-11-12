use crate::error::{Result, ServerError};
use candle_core::Device;
use candle_nn::VarBuilder;
use std::path::Path;

/// Load model weights from safetensors or other formats
pub fn load_weights(path: &str, device: &Device) -> Result<VarBuilder> {
    let model_path = Path::new(path);

    if !model_path.exists() {
        return Err(ServerError::ModelNotFound(format!(
            "Model path does not exist: {}",
            path
        )));
    }

    // Check if it's a directory with multiple safetensors files
    if model_path.is_dir() {
        load_from_directory(model_path, device)
    } else if path.ends_with(".safetensors") {
        load_single_file(model_path, device)
    } else {
        Err(ServerError::ModelLoading(format!(
            "Unsupported model format. Expected .safetensors file or directory containing safetensors files"
        )))
    }
}

fn load_single_file(path: &Path, device: &Device) -> Result<VarBuilder> {
    tracing::info!("Loading model from single file: {:?}", path);

    unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, device)
            .map_err(|e| {
                ServerError::ModelLoading(format!("Failed to load safetensors: {}", e))
            })
    }
}

fn load_from_directory(dir: &Path, device: &Device) -> Result<VarBuilder> {
    tracing::info!("Loading model from directory: {:?}", dir);

    // Find all .safetensors files in the directory
    let mut safetensor_files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| ServerError::Io(e))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    if safetensor_files.is_empty() {
        return Err(ServerError::ModelLoading(
            "No .safetensors files found in directory".to_string(),
        ));
    }

    // Sort files for consistent loading order
    safetensor_files.sort();

    tracing::info!(
        "Found {} safetensors files: {:?}",
        safetensor_files.len(),
        safetensor_files
    );

    // Load all files using memory-mapped IO for efficiency
    unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, candle_core::DType::F32, device)
            .map_err(|e| {
                ServerError::ModelLoading(format!("Failed to load safetensors: {}", e))
            })
    }
}

/// Download a model from Hugging Face Hub
pub async fn download_from_hf(
    repo_id: &str,
    revision: Option<&str>,
    cache_dir: Option<&Path>,
) -> Result<String> {
    use hf_hub::api::tokio::Api;

    tracing::info!("Downloading model from Hugging Face: {}", repo_id);

    let mut api = Api::new().map_err(|e| {
        ServerError::ModelLoading(format!("Failed to initialize HF Hub API: {}", e))
    })?;

    if let Some(dir) = cache_dir {
        api = api.with_cache_dir(dir.to_path_buf());
    }

    let mut repo = api.model(repo_id.to_string());
    if let Some(rev) = revision {
        repo = repo.revision(rev.to_string());
    }

    let model_path = repo.get("model.safetensors").await.map_err(|e| {
        ServerError::ModelLoading(format!("Failed to download model: {}", e))
    })?;

    Ok(model_path
        .parent()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_validation() {
        let result = load_weights("/nonexistent/path", &Device::Cpu);
        assert!(result.is_err());
    }
}
