/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use cxx::ExternType;
use cxx::type_id;

#[cfg(not(any(use_rocm, use_xpu)))]
mod extern_types {
    use super::*;

    /// SAFETY: bindings
    unsafe impl ExternType for CUstream_st {
        type Id = type_id!("CUstream_st");
        type Kind = cxx::kind::Opaque;
    }

    /// SAFETY: bindings
    unsafe impl ExternType for ncclComm {
        type Id = type_id!("ncclComm");
        type Kind = cxx::kind::Opaque;
    }
}

#[cfg(use_rocm)]
mod extern_types {
    use super::inner::ihipStream_t;
    use super::inner::ncclComm;
    use super::*;

    /// SAFETY: bindings
    /// Note: HIP uses ihipStream_t as the opaque type behind hipStream_t pointer
    unsafe impl ExternType for ihipStream_t {
        type Id = type_id!("ihipStream_t");
        type Kind = cxx::kind::Opaque;
    }

    /// SAFETY: bindings
    unsafe impl ExternType for ncclComm {
        type Id = type_id!("ncclComm");
        type Kind = cxx::kind::Opaque;
    }
}
// TODO CHECK WHERE IT IS NEEDED
// #[cfg(use_xpu)]
// mod extern_types {
//     use super::inner::ncclComm;
//     use super::inner::xpuStreamOpaque;
//     use super::*;

//     /// SAFETY: bindings
//     /// Note: XPU uses xpuStreamOpaque as the opaque type behind xpuStream_t pointer
//     unsafe impl ExternType for xpuStreamOpaque {
//         type Id = type_id!("xpuStreamOpaque");
//         type Kind = cxx::kind::Opaque;
//     }

//     /// SAFETY: bindings
//     unsafe impl ExternType for ncclComm {
//         type Id = type_id!("ncclComm");
//         type Kind = cxx::kind::Opaque;
//     }
// }

// When building with cargo, this is actually the lib.rs file for a crate.
// Include the generated bindings.rs and suppress lints.
#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
mod inner {
    use serde::Deserialize;
    use serde::Deserializer;
    use serde::Serialize;
    use serde::Serializer;
    use serde::ser::SerializeSeq;
    #[cfg(cargo)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    // This type is manually defined instead of generated because we want to derive
    // Serialize/Deserialize on it.
    // oneCCL unique IDs are 4096 bytes; NCCL/RCCL unique IDs are 128 bytes.
    #[cfg(use_xpu)]
    pub const NCCL_UNIQUE_ID_BYTES: usize = 4096;
    #[cfg(not(use_xpu))]
    pub const NCCL_UNIQUE_ID_BYTES: usize = 128;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Serialize, Deserialize)]
    pub struct ncclUniqueId {
        // Custom serializer required, as serde does not provide a built-in
        // implementation of serialization for large arrays.
        #[serde(
            serialize_with = "serialize_array",
            deserialize_with = "deserialize_array"
        )]
        pub internal: [::std::os::raw::c_char; NCCL_UNIQUE_ID_BYTES],
    }

    fn deserialize_array<'de, D>(
        deserializer: D,
    ) -> Result<[::std::os::raw::c_char; NCCL_UNIQUE_ID_BYTES], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<::std::os::raw::c_char> = Deserialize::deserialize(deserializer)?;
        vec.try_into().map_err(|v: Vec<::std::os::raw::c_char>| {
            serde::de::Error::custom(format_args!(
                "expected an array of length {}, got {}",
                NCCL_UNIQUE_ID_BYTES,
                v.len()
            ))
        })
    }

    fn serialize_array<S>(
        array: &[::std::os::raw::c_char; NCCL_UNIQUE_ID_BYTES],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(NCCL_UNIQUE_ID_BYTES))?;
        for element in array {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

// Export all inner bindings for CUDA, ROCm, and XPU builds
pub use inner::*;

// For ROCm: also export compatibility aliases that map CUDA names to HIP
#[cfg(use_rocm)]
pub use self::rocm_compat::*;

#[cfg(use_rocm)]
#[allow(non_camel_case_types)]
mod rocm_compat {
    use super::inner;

    // ROCm/HIP compatibility layer
    //
    // Hipify converts CUDA APIs to HIP in C++ code, causing bindgen to generate HIP types.
    // These aliases map CUDA names back to their HIP equivalents for Rust code compatibility.
    pub type cudaError_t = inner::hipError_t;
    pub type cudaStream_t = inner::hipStream_t;
    pub type CUstream_st = inner::ihipStream_t;

    // Function aliases - hipify converts cudaSetDevice -> hipSetDevice, etc.
    pub use inner::hipSetDevice as cudaSetDevice;
    pub use inner::hipStreamSynchronize as cudaStreamSynchronize;
}

// TODO CHECK WHERE IT IS NEEDED
// // For XPU: export compatibility aliases that map CUDA names to XPU equivalents
// #[cfg(use_xpu)]
// pub use self::xpu_compat::*;

// #[cfg(use_xpu)]
// #[allow(non_camel_case_types)]
// mod xpu_compat {
//     use super::inner;

//     // XPU/oneCCL compatibility layer
//     //
//     // On XPU, streams are opaque pointers to SYCL queues.
//     // These aliases map CUDA names to XPU equivalents for Rust code compatibility.
//     pub type cudaError_t = inner::xpuError_t;
//     pub type cudaStream_t = inner::xpuStream_t;
//     pub type CUstream_st = inner::xpuStreamOpaque;

//     // Function aliases - map CUDA runtime functions to XPU equivalents
//     pub use inner::xpuSetDevice as cudaSetDevice;
//     pub use inner::xpuStreamSynchronize as cudaStreamSynchronize;
// }

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::*;

    #[test]
    fn sanity() {
        // SAFETY: testing bindings
        unsafe {
            let mut version = MaybeUninit::<i32>::uninit();
            let result = ncclGetVersion(version.as_mut_ptr());
            assert_eq!(result.0, 0);
        }
    }
}
