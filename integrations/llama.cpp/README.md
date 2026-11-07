# ggml-pcuda: pcuda as a llama.cpp (ggml) backend

Copy of the backend developed in a llama.cpp tree (branch `pcuda-backend` of the local fork).

To add it to a llama.cpp checkout:

```powershell
Copy-Item -Recurse integrations\llama.cpp\ggml\* <llama.cpp>\ggml\
git -C <llama.cpp> apply <this repo>\integrations\llama.cpp\ggml-pcuda-hooks.patch
cmake -S <llama.cpp> -B build-pcuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_PCUDA=ON -DGGML_PCUDA_ROOT=<this repo>
cmake --build build-pcuda --target llama-server llama-bench
```

The hook patch adds the `GGML_PCUDA` / `GGML_PCUDA_ROOT` options, `ggml_add_backend(PCUDA)` and the
static registration. See the main README ("ggml backend prototype") for what the backend does
and how fast it is.
