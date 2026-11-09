# ggml-pcuda: pcuda as a llama.cpp (ggml) backend

Copy of the backend developed in a llama.cpp tree (branch `pcuda-backend` of the local fork).

To add it to a llama.cpp checkout:

```powershell
Copy-Item -Recurse integrations\llama.cpp\ggml\* <llama.cpp>\ggml\
git -C <llama.cpp> apply <this repo>\integrations\llama.cpp\ggml-pcuda-hooks.patch
cmake -S <llama.cpp> -B build-pcuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_PCUDA=ON -DGGML_PCUDA_ROOT=<this repo>
cmake --build build-pcuda --target llama-server llama-bench
```

The hook patch adds the `GGML_PCUDA` / `GGML_PCUDA_ROOT` options, `ggml_add_backend(PCUDA)`, the
static registration, and a change in `src/llama-context.cpp`. That change keeps each CPU layer's norms on an ACCEL backend that
supports them, instead of pinning them to the CPU, so a layer is not split around every norm. A
change in `src/llama-model.cpp` lets ACCEL devices turn off mmap in the default load mode: pcuda
copies the weights into its own memory, so a mapped file would keep a second copy resident. For
attention in pcuda, `src/llama-kv-cache.cpp` puts a CPU layer's KV cache in an ACCEL device's host
buffer type when it offers one (pcuda memory the CPU backend can also use), and the flash-attention
check in `src/llama-context.cpp` accepts an ACCEL device running a CPU layer's attention. See the main README ("ggml backend prototype") for what the backend does
and how fast it is.
