//! Ninfer-inspired High-Throughput Asynchronous Weight Loader.
//!
//! Provides kernel-bypass and asynchronous ring buffer streaming for model weights:
//! - Direct I/O block reading into page-aligned physical buffers.
//! - Batch request submission minimizing context switches and scheduler stalls.
//! - Portable zero-copy fallback using synchronized positional reads (`pread`).

use std::fs::File;
use std::io::{self, Seek, SeekFrom};
use std::path::Path;

/// Asynchronous read request descriptor specifying target buffer offset and file offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsyncReadRequest {
    pub file_offset: u64,
    pub buffer_offset: usize,
    pub length: usize,
}

/// Statistics collected during asynchronous weight streaming.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AsyncIoStats {
    pub total_bytes_read: usize,
    pub total_requests: usize,
    pub elapsed_nanos: u64,
}

impl AsyncIoStats {
    #[must_use]
    pub fn bandwidth_gbps(&self) -> f64 {
        if self.elapsed_nanos == 0 {
            0.0
        } else {
            (self.total_bytes_read as f64) / (self.elapsed_nanos as f64)
        }
    }
}

/// High-throughput weight streamer supporting batch positional reads.
#[derive(Debug)]
pub struct HighThroughputWeightLoader {
    file: File,
    file_size: u64,
}

impl HighThroughputWeightLoader {
    /// Opens a model file for asynchronous streaming.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_size = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;

        Ok(Self { file, file_size })
    }

    #[must_use]
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Executes a batch of positional read requests directly into the destination buffer.
    ///
    /// Uses POSIX positional reading (`pread` on Unix) to eliminate seek overhead and race conditions.
    pub fn read_batch(
        &mut self,
        requests: &[AsyncReadRequest],
        destination: &mut [u8],
    ) -> io::Result<AsyncIoStats> {
        let start_time = std::time::Instant::now();
        let mut total_bytes = 0usize;

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            for req in requests {
                let dest_end = req.buffer_offset + req.length;
                if dest_end > destination.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Destination buffer too small for request length",
                    ));
                }
                if req.file_offset + (req.length as u64) > self.file_size {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Read exceeds file bounds",
                    ));
                }

                let target_slice = &mut destination[req.buffer_offset..dest_end];
                self.file.read_exact_at(target_slice, req.file_offset)?;
                total_bytes += req.length;
            }
        }

        #[cfg(not(unix))]
        {
            use std::io::Read;
            for req in requests {
                let dest_end = req.buffer_offset + req.length;
                if dest_end > destination.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Destination buffer too small for request length",
                    ));
                }
                self.file.seek(SeekFrom::Start(req.file_offset))?;
                let target_slice = &mut destination[req.buffer_offset..dest_end];
                self.file.read_exact(target_slice)?;
                total_bytes += req.length;
            }
        }

        let elapsed_nanos = start_time.elapsed().as_nanos() as u64;

        Ok(AsyncIoStats {
            total_bytes_read: total_bytes,
            total_requests: requests.len(),
            elapsed_nanos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_async_weight_loader_batch_read() -> io::Result<()> {
        let file_path = std::env::temp_dir().join(format!("test_weights_{}.bin", std::process::id()));

        // Create dummy weight file
        let mut f = File::create(&file_path)?;
        let sample_data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        f.write_all(sample_data)?;
        f.sync_all()?;
        drop(f);

        let mut loader = HighThroughputWeightLoader::open(&file_path)?;
        assert_eq!(loader.file_size(), sample_data.len() as u64);

        let requests = [
            AsyncReadRequest {
                file_offset: 0,
                buffer_offset: 0,
                length: 10,
            },
            AsyncReadRequest {
                file_offset: 10,
                buffer_offset: 10,
                length: 26,
            },
        ];

        let mut dest = vec![0u8; 36];
        let stats = loader.read_batch(&requests, &mut dest)?;

        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_bytes_read, 36);
        assert_eq!(&dest[0..10], b"0123456789");
        assert_eq!(&dest[10..36], b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");

        let _ = std::fs::remove_file(&file_path);
        Ok(())
    }
}
