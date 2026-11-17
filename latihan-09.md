# Latihan Mandiri di Lab — Modul 09

**Nama:** Benget Sidabutar  
**NIM:** 123450047

## Prediksi

1. *Parameter:* keempat model memiliki jumlah parameter yang **sama**. Setiap head memakai potongan berdimensi $d_k=d_{\text{model}}/h$, sehingga $W_Q,W_K,W_V,W_O$ tetap berukuran $d_{\text{model}}\times d_{\text{model}}$ dan $h$ tidak muncul pada rumus $4d^2+4d$. Perkiraan: 40.400 untuk multi-head attention dan 76.075 untuk satu layer encoder, untuk semua $h$.
2. *Waktu:* parameter sama tidak berarti waktu sama. Perkiraan: waktu forward **naik** seiring $h$ (urutan $h=1\lesssim h=2<h=4<h=10$) dan selisihnya **melebar** saat $n$ membesar, karena matriks bobot attention berukuran $B\times h\times n\times n$ sehingga jumlah entri softmax tumbuh $\propto h\cdot n^2$, sedangkan FLOPs perkalian matriks tetap ($h\cdot n^2\cdot d_k=n^2\cdot d_{\text{model}}$).

## Kode

```python
# Latihan mandiri: n_head in (1, 2, 4, 10) pada D_MODEL = 100 tetap, tanpa training.
NHEADS = (1, 2, 4, 10)

# --- (a) Parameter ---------------------------------------------------------
model_head, baris = {}, []
for h in NHEADS:
    assert D_MODEL % h == 0, 'D_MODEL harus habis dibagi n_head'
    seed_everything(BASE_SEED)
    mdl = TransformerClassifier(nhead=h)
    model_head[h] = mdl
    baris.append({
        'n_head': h,
        'd_k': D_MODEL // h,
        'param_mha': sum(p.numel() for p in mdl.layer.self_attn.parameters()),
        'param_layer': sum(p.numel() for p in mdl.layer.parameters()),
        'param_total': sum(p.numel() for p in mdl.parameters()),
    })
tabel_head = pd.DataFrame(baris)
RUMUS_MHA = 4 * D_MODEL ** 2 + 4 * D_MODEL        # W_Q, W_K, W_V, W_O + empat bias; tanpa h
print('rumus 4d^2 + 4d =', RUMUS_MHA)
assert (tabel_head['param_mha'] == RUMUS_MHA).all(), 'parameter MHA harus sama dengan rumus'
assert tabel_head['param_layer'].nunique() == 1, 'parameter layer harus sama untuk semua h'
assert tabel_head.loc[tabel_head.n_head == N_HEAD, 'param_layer'].item() == parameter_layer(D_MODEL, FF)

# --- (b) Waktu forward -----------------------------------------------------
# n = 60 diukur tiga putaran (median per putaran sudah dari waktu_forward) lalu diambil
# mediannya agar tahan terhadap lonjakan beban CPU; n = 120 untuk melihat tren terhadap n.
for n_ in (60, 120):
    putaran = np.array([[waktu_forward(model_head[h], n_) for h in NHEADS] for _ in range(3)])
    tabel_head[f'ms_n{n_}'] = np.median(putaran, axis=0)
    tabel_head[f'rasio_n{n_}'] = tabel_head[f'ms_n{n_}'] / tabel_head[f'ms_n{n_}'].iloc[0]
# Entri matriks bobot attention per forward pass: B * h * n * n  (alasan yang diuji)
for n_ in (60, 120):
    tabel_head[f'entri_bobot_n{n_}'] = 64 * tabel_head['n_head'] * n_ * n_

print(tabel_head.to_string(index=False))
```

## Keluaran

```
rumus 4d^2 + 4d = 40400
 n_head  d_k  param_mha  param_layer  param_total    ms_n60  rasio_n60   ms_n120  rasio_n120  entri_bobot_n60  entri_bobot_n120
      1  100      40400        76075      1246879  6.791211   1.000000 13.938894    1.000000           230400            921600
      2   50      40400        76075      1246879  6.749677   0.993884 17.876542    1.282494           460800           1843200
      4   25      40400        76075      1246879  8.013947   1.180047 27.458894    1.969948           921600           3686400
     10   10      40400        76075      1246879 13.173190   1.939741 57.917799    4.155122          2304000           9216000
```

Pengukuran waktu dilakukan pada CPU (2 thread), PyTorch 2.15, batch 64, median dari tiga putaran; angka milidetik bergantung pada mesin, sedangkan jumlah parameter tidak.

## Hasil dan perbandingan dengan prediksi

- **Parameter (sesuai prediksi).** Keempat model identik: multi-head attention 40.400, satu layer encoder 76.075, model total 1.246.879 (termasuk embedding). $W_Q,W_K,W_V$ memetakan $d_{\text{model}}\to h\cdot d_k=d_{\text{model}}$ dan $W_O$ memetakan kembali, sehingga tiap matriks tetap $d_{\text{model}}^2$ bobot ditambah bias. Total $4d^2+4d$ tidak memuat $h$: menambah head hanya membagi ulang dimensi menjadi potongan $d_k=d_{\text{model}}/h$ yang lebih sempit.
- **Waktu (sesuai prediksi, dengan satu catatan).** Pada $n=60$ waktu naik dari 6,8 ms ($h=1$) menjadi 13,2 ms ($h=10$), sekitar 1,9×; pada $n=120$ dari 13,9 ms menjadi 57,9 ms, sekitar 4,2×. Urutan naik dan selisih melebar terhadap $n$ sesuai prediksi, tetapi $h=1$ dan $h=2$ praktis sama pada $n=60$ (selisih di bawah derau). Angka milidetik bergantung pada mesin dan berbeda sedikit pada tiap run.
- **Alasan yang dapat diuji.** FLOPs perkalian matriks sama untuk semua $h$, tetapi matriks bobot attention berukuran $B\times h\times n\times n$, sehingga entri yang di-softmax tumbuh $\propto h\cdot n^2$ (kolom `entri_bobot`: 230.400 → 2.304.000 pada $n=60$). Bila alasan ini benar, rasio waktu $h=10$ terhadap $h=1$ harus naik dengan $n$; itu teramati (1,9× → 4,2×). Waktu naik kurang dari 10× karena embedding, feed-forward, dan LayerNorm tidak bergantung pada $h$. Penjelasan ini konsisten dengan data tetapi belum dibuktikan dengan profiler.
