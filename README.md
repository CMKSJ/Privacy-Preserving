# Privacy-Preserving (Docker-only)

This project is configured to run fully inside Docker.

## Commands
From `course/SE6019/Privacy-Preserving`:

1. Build image
```bash
make docker-build
```

2. Train HE-friendly model
```bash
make docker-train
```

3. Single encrypted inference demo
```bash
make docker-he-single
```

4. Batch encrypted evaluation
```bash
make docker-he-batch
```

5. Compare plaintext vs ciphertext logits
```bash
make docker-compare
```

## Notes
- Outputs (model/data) are written into the mounted workspace, not inside ephemeral container storage.
- This setup avoids local dependency installation.
