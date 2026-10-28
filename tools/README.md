# tools/ — offline deployment, on-site acceptance, model export

Operational tooling that sits outside the shipping package. Nothing here is imported by
`jasna` at runtime, so it does not affect the build or the runtime import graph.

## 1. What runs where

| Tool | Runs in |
| --- | --- |
| `jasna_bundle_check.py` | any Python 3.8+; needs neither the product runtime nor a license |
| `jasna_deploy_verify.py` | inside the frozen (Nuitka) release process |
| `decrypt_acceptance.sh` | host shell; drives the frozen binary as a subprocess |
| `jasna --export-models ...` | inside the frozen release process (implemented by `jasna/protected_io.py`) |

The split follows one hard constraint: `.enc` payloads can only be opened by the frozen
build. `jasna.model_files.open_tensors` dispatches on `jasna._frozen.is_frozen()` — source
mode takes the plaintext `safe_open` path, and `jasna.protection.protected_tensors` exists
only inside the compiled release. Anything that reads ciphertext therefore has to run in
that process.

## 2. The `.enc` container

Verified chunk-by-chunk against real files on a production host:

```text
file  = [29-byte header] + N x ( [4-byte chunk prefix] + [ciphertext chunk] )
header = "JASNAENC1"(9) + nonce(8) + 12 reserved bytes
```

Each `<name>.enc` has a sibling `<name>.enc.manifest.json` carrying per-chunk ciphertext
lengths and SHA-256 values plus a 64-byte signature. Because
`29 + 4*N + sum(lengths)` equals the file size exactly, chunk-level integrity can be
checked **without decrypting**. Signature verification needs the product public key and
therefore only works inside the frozen process.

## 3. jasna_bundle_check.py

Validates a model directory against the published manifest and each file's own manifest.

```bash
python3 tools/jasna_bundle_check.py --root /data/jasna/model_weights --report bundle.json
```

| Option | Effect |
| --- | --- |
| `--root` | model directory, defaults to `.` |
| `--mode` | `auto` prefers `.enc` then plaintext; `enc` / `plain` accept one form only |
| `--size-tolerance` | allowed byte delta; use 256 when mixing v0.11.3 and v0.12.0 |
| `--no-deep` | skip whole-file SHA-256, size checks only |
| `--verify-chunks` | verify every ciphertext chunk against the manifest (strongest non-decrypting check) |
| `--trust-manifest` | downgrade a published-digest mismatch to WARN when the manifest is self-consistent |
| `--allow-container` | accept overlay and similar filesystems as WARN instead of FAIL |

Checks: manifest structure and chunk hashes, published size/digest, the eight model
variants declared by `bundle_names()`, filesystem type, unlisted `.enc` assets, and
plaintext/ciphertext mixes of the same published name.

## 4. jasna_deploy_verify.py

Environment and deployment acceptance. Must be embedded in the release or invoked by it,
because `_core` locates the Jasna folder via `dirname(/proc/self/exe)` and the frozen
build has no separate interpreter to borrow.

```bash
# capture a baseline once, on the build/release machine
python3 tools/jasna_deploy_verify.py record --root /opt/jasna --out baseline.json

# on site
jasna verify-deployment --root /opt/jasna --license /etc/jasna/license.bin \
    --baseline baseline.json --report /tmp/acceptance.json --strict
```

Two layers: product-independent gates (CPython ABI, architecture, filesystem type,
`/proc/self/uid_map`, manifest digests, permissions, ELF validity) and product checks
(`extension_verified`, `machine_key` + `authorize`, `check_ltx_worker_mappings`,
`create_code_cache`, `launch_ltx_worker`). A product check that cannot run reports SKIP
rather than guessing.

## 5. decrypt_acceptance.sh

End-to-end proof that the encrypted weights really decrypt: runs one short restoration
span and asserts the evidence.

```bash
bash tools/decrypt_acceptance.sh <gpu-index> [input.mp4]
```

Assertions: exit code 0, output video larger than 1 MB, the log contains
`blocks 48/48 resident`, and the log names the encrypted transformer file. Writes
`decrypt_acceptance.json` next to the model weights.

First run on a given GPU compiles the TensorRT detection engine (about 12 minutes);
subsequent runs take roughly 70 seconds. `--device cuda:N` alone does not work for the
compile step — TensorRT's builder is bound to device 0, so select the GPU with
`CUDA_VISIBLE_DEVICES` and keep `--device cuda:0`. Release and model paths inside the
script are site-specific; adjust them before reuse.

## 6. Exit codes

| Code | Meaning |
| --- | --- |
| 0 | passed (warnings allowed) |
| 1 | failed |
| 2 | inconclusive (a required check was SKIP) |
| 3 | usage error or refused to run |

## 7. Verified behaviour

`jasna_bundle_check.py`: empty directory, complete set, single variant, mixed layout,
size mismatch, digest mismatch, tolerance path, and 225 real chunks verified in 35 s.
`jasna_deploy_verify.py`: unchanged, tampered, added, missing, world-writable, import
error mapping, worker timeout. `decrypt_acceptance.sh`: 76 s cold-cached run, four
assertions green.

## 8. Caveats

- The bundle check's published digests are pinned to the release that generated them. A
  transformer built by a different release is legitimate but will differ; reconcile the
  version before treating it as tampering.
- `jasna --export-models` writes unprotected plaintext. It requires a valid license and an
  output directory with mode `0700`, and it refuses to run without
  `--confirm-unprotected-output`. Do not ship the plaintext it produces.
