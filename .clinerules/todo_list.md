### TODO

- **Reduce download latency during analysis & steering**  
  The biggest performance bottleneck is downloading intermediate data (e.g. activation vectors) from the nnsight server during:
  1. **Layer effect analysis (probing)**  
  2. **Steer evaluation (steering)**  

  **Proposed fix:**  
  - Only download (and call `.save()`) when the raw vectors are actually needed—namely, during train_vectors.py.  
  - In steering runs, use activations (or another lightweight proxy) in memory and defer saving/downloading until the final effect is computed.  
  - This avoids unnecessary vector transfers and significantly cuts down end-to-end latency.
