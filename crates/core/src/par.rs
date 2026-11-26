// Data parallelism when the `parallel` feature (rayon) is on; sequential otherwise (WebAssembly).
pub fn chunks_mut_enumerate<T: Send, F>(data: &mut [T], size: usize, f: F)
where
    F: Fn((usize, &mut [T])) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(size).enumerate().for_each(f);
    }
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(size).enumerate().for_each(f);
}
