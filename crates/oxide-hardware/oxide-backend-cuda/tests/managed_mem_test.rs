use oxide_backend_cuda::driver::{cudaError_t, CudaDeviceBuffer, CUDA_SUCCESS, CUDA_MEMCPY_DEFAULT, CUDA_MEMCPY_HOST_TO_DEVICE};
use std::ffi::c_void;
use std::ptr;

#[test]
fn test_managed_memory_behavior() {
    let size = 32 * 1024 * 1024; // 32MB
    let mut d_ptr: *mut c_void = ptr::null_mut();
    
    unsafe extern "C" {
        fn cudaMallocManaged(dev_ptr: *mut *mut c_void, size: usize, flags: u32) -> cudaError_t;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: u32) -> cudaError_t;
        fn cudaFree(ptr: *mut c_void) -> cudaError_t;
        fn cudaDeviceSynchronize() -> cudaError_t;
    }

    let status = unsafe { cudaMallocManaged(&raw mut d_ptr, size, 1) };
    println!("cudaMallocManaged status: {status}, ptr: {:?}", d_ptr);
    assert_eq!(status, CUDA_SUCCESS);
    assert!(!d_ptr.is_null());

    let host_data = vec![0x42u8; size];
    
    // Test 1: cudaMemcpy with CUDA_MEMCPY_DEFAULT
    let copy_status = unsafe { cudaMemcpy(d_ptr, host_data.as_ptr().cast(), size, CUDA_MEMCPY_DEFAULT) };
    println!("cudaMemcpy DEFAULT status: {copy_status}");

    // Test 2: direct CPU write
    unsafe {
        std::ptr::copy_nonoverlapping(host_data.as_ptr(), d_ptr.cast::<u8>(), 1024);
    }
    println!("Direct CPU copy succeeded!");

    // Test 3: synchronize
    let sync_status = unsafe { cudaDeviceSynchronize() };
    println!("cudaDeviceSynchronize status: {sync_status}");

    // Test 5: cudaMemcpyAsync on managed memory with stream
    let mut stream: *mut c_void = ptr::null_mut();
    unsafe extern "C" {
        fn cudaStreamCreate(stream: *mut *mut c_void) -> cudaError_t;
        fn cudaStreamDestroy(stream: *mut c_void) -> cudaError_t;
        fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: u32, stream: *mut c_void) -> cudaError_t;
    }
    let st_status = unsafe { cudaStreamCreate(&raw mut stream) };
    println!("cudaStreamCreate status: {st_status}");

    let mut managed_buf: *mut c_void = ptr::null_mut();
    let m_status = unsafe { cudaMallocManaged(&raw mut managed_buf, size, 1) };
    println!("cudaMallocManaged status: {m_status}, ptr: {:?}", managed_buf);

    // Test 6: mmap source to managed memory destination via cudaMemcpyAsync
    let temp_path = "/tmp/test_cuda_mmap.bin";
    std::fs::write(temp_path, &vec![0x77u8; size]).unwrap();
    let file = std::fs::File::open(temp_path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    println!("mmap ptr: {:?}, len: {}", mmap.as_ptr(), mmap.len());

    let mmap_copy_status = unsafe {
        cudaMemcpyAsync(
            managed_buf,
            mmap.as_ptr().cast(),
            mmap.len(),
            CUDA_MEMCPY_DEFAULT,
            stream,
        )
    };
    println!("cudaMemcpyAsync from mmap to managed memory status: {mmap_copy_status}");
    let mmap_sync = unsafe { cudaDeviceSynchronize() };
    println!("cudaDeviceSynchronize after mmap copy status: {mmap_sync}");
    let _ = std::fs::remove_file(temp_path);

    unsafe {
        cudaStreamDestroy(stream);
        cudaFree(managed_buf);
    }
}
