/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! oneAPI installation detection and utilities
//!
//! This module provides functionality for detecting oneAPI installations,
//! and validating versions.

use std::path::Path;

use crate::BuildError;
use crate::get_env_var_with_rerun;

/// Validate oneAPI installation exists and return oneAPI home path
///
/// Checks for oneAPI installation through:
/// 1. ONEAPI_PATH environment variable
/// 2. Default location /opt/intel/oneapi
/// 3. Finding dpcpp in PATH and resolving symlinks
pub fn validate_oneapi_installation() -> Result<String, BuildError> {
    // Try ONEAPI_PATH environment variable first
    if let Ok(oneapi_path) = get_env_var_with_rerun("ONEAPI_PATH") {
        if Path::new(&oneapi_path).join("/compiler/latest/bin/dpcpp").exists() {
            return Ok(oneapi_path);
        }
    }

    // TODO: implement other way of oneapi discovery
    // // Try default location /opt/oneapi (handles versioned installs like /opt/oneapi-7.1.1 via symlink)
    // let default_oneapi = "/opt/intel/oneapi";
    // if Path::new(default_oneapi).join("bin/hipcc").exists() {
    //     // Resolve symlink to get actual versioned path if it exists
    //     if let Ok(canonical) = fs::canonicalize(default_oneapi) {
    //         return Ok(canonical.to_string_lossy().to_string());
    //     }
    //     return Ok(default_rocm.to_string());
    // }

    // // Try finding hipcc in PATH and resolving symlinks
    // if let Ok(hipcc_path) = which("hipcc") {
    //     // Resolve symlinks to get the real path
    //     if let Ok(real_hipcc) = fs::canonicalize(&hipcc_path) {
    //         if let Some(rocm_home) = real_hipcc.parent().and_then(|p| p.parent()) {
    //             return Ok(rocm_home.to_string_lossy().to_string());
    //         }
    //     }
    // }

    Err(BuildError::PathNotFound("oneAPI installation".to_string()))
}

/// Get oneAPI version from installation
///
/// Returns (major, minor) version tuple.
pub fn get_oneapi_version(oneapi_home: &str) -> Result<(u32, u32), BuildError> {
    return Ok((0, 0)); // TODO: implement oneAPI version detection

    // let version_file = Path::new(oneapi_home).join(".info/version");

    // if let Ok(content) = fs::read_to_string(&version_file) {
    //     // Parse version like "6.0.2" or "7.0.0"
    //     let parts: Vec<&str> = content.trim().split('.').collect();
    //     if parts.len() >= 2 {
    //         if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
    //             // Enforce ROCm 7.0+ requirement
    //             if major < 7 {
    //                 return Err(BuildError::CommandFailed(format!(
    //                     "ROCm {}.{} detected, but ROCm 7.0+ is required",
    //                     major, minor
    //                 )));
    //             }
    //             return Ok((major, minor));
    //         }
    //     }
    // }

    // Err(BuildError::PathNotFound("ROCm version file".to_string()))
}

/// Get oneAPI library directory
pub fn get_oneapi_lib_dir() -> Result<String, BuildError> {
    let oneapi_home = validate_oneapi_installation()?;
    let lib_path = Path::new(&oneapi_home).join("lib");

    if lib_path.exists() {
        Ok(lib_path.to_string_lossy().to_string())
    } else {
        Err(BuildError::PathNotFound("oneAPI lib directory".to_string()))
    }
}
