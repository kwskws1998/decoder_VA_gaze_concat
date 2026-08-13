# Legacy and reference material

Nothing in this directory is imported by the active Qwen VA pipeline.

- `original_va_model/` preserves the 2023 encoder-based VA implementation.
- `gaze_reward_reference/` preserves the upstream GazeReward/GazeConcat code
  used as architectural reference. Its original LGPL license remains with it.
- `paper_evidence/` contains the paper PDF and page-render evidence used during
  protocol review.

Do not install dependencies or run experiments from this directory. Active
training code is exclusively under `../va_model_code/`.
