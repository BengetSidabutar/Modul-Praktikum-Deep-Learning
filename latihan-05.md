# Modul 5: Convolutional Neural Network Dasar

**Nama:** Benget Sidabutar
**NIM:** 123450047
**Kelas:** RB
**Tanggal:** YYYY-MM-DD

Simpan berkas ini sebagai `M05_NIM.ipynb` sebelum mulai mengerjakan.

## Petunjuk

Notebook ini memiliki tiga jenis bagian. Tanda di akhir judul menunjukkan jenisnya.

| Tanda | Bagian | Yang Anda lakukan |
|:---|:---|:---|
| **Terbimbing** | Pre-lab, A–E | Jawaban lengkap Prosedur Praktikum Terbimbing sudah tersedia, termasuk pelatihan CNN baseline. Jalankan, baca kodenya, lalu cocokkan keluarannya dengan tab **Materi** modul. |
| **Latihan Mandiri di Lab** | Latihan | Dikerjakan sendiri di laboratorium setelah bagian terbimbing. |
| **Tugas Individu** | T1–T4 dan Pertanyaan analisis | Dikerjakan di luar sesi dan dikumpulkan lewat Tugas. |

1. Isi identitas di atas dan `NIM` pada sel pertama. Hanya itu isian pada bagian terbimbing. `DATASET = 'fashion'` sudah diatur untuk laptop tanpa GPU.
2. Tekan **Siapkan dataset di laptop** di halaman modul agar Fashion-MNIST tersedia di `data/raw`. Notebook tidak mengunduh apa pun.
3. Tekan **Restart**, lalu **Run all**. Seluruh sel terbimbing berjalan di kernel Deep Learning (PyTorch CPU) dan berhenti di sel Latihan pertama yang masih berisi `TODO`.
4. Keluaran yang tidak bergantung pada NIM, misalnya hitungan cross-correlation, gradien, dan tabel anatomi, harus sama persis dengan naskah. Tensor acak, split data, dan hasil training bergantung pada seed NIM, sedangkan waktu bergantung pada perangkat.
5. Pada bagian Latihan dan Tugas, ganti seluruh penanda `TODO` dan jangan menghapus sel pemeriksaan `cek(...)` atau `assert`. Implementasi manual hanya boleh memakai indexing, perkalian elemen, penjumlahan, `F.pad`, dan `torch.flip`; verifikasi memakai `torch.float64`. Validation set dipakai untuk memilih model, dan test set hanya dipakai satu kali pada Tugas T4.
6. Sebelum mengumpulkan, jalankan **Restart** lalu **Run all** sampai seluruh sel selesai tanpa galat. Luaran: `M05_NIM.ipynb`, `M05_NIM.pdf`, dan `M05_NIM_metrics.csv`; sebutkan pada laporan bahwa dataset yang dipakai Fashion-MNIST.

```python
import platform
import random
import statistics
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import Markdown, display
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, FashionMNIST

# Kernel Workbench menangkap figure di akhir setiap sel; plt.show() dipertahankan
# agar notebook tetap berjalan di Jupyter biasa, tanpa peringatan backend Agg.
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')

DATASET = 'fashion'      # Workbench (CPU): Fashion-MNIST dari data/raw; 'cifar10' hanya bila datanya tersedia
NIM = '123450047'  # Ganti dengan NIM lengkap Anda, misalnya '120140167'.
assert NIM.isdigit() and len(NIM) >= 4, 'Isi NIM dengan angka sebelum melanjutkan.'
SEED = int(str(NIM)[-4:]) if str(NIM).isdigit() else 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cek(nama, hasil, acuan):
    """Sel pemeriksaan: shape harus sama dan nilai lolos torch.allclose."""
    assert hasil is not None and acuan is not None, f'{nama}: nilai belum diisi (masih None)'
    hasil = torch.as_tensor(hasil.detach() if torch.is_tensor(hasil) else hasil, dtype=DT)
    acuan = torch.as_tensor(acuan.detach() if torch.is_tensor(acuan) else acuan, dtype=DT)
    ok = hasil.shape == acuan.shape and torch.allclose(hasil, acuan)
    print(f'[{"OK" if ok else "GAGAL"}] {nama}: shape {tuple(hasil.shape)}')
    assert ok, f'{nama} tidak cocok dengan acuan'

def ukur(fungsi, ulang=10):
    """Median waktu eksekusi `fungsi` dalam detik setelah satu pemanasan."""
    fungsi()
    waktu = []
    for _ in range(ulang):
        mulai = time.perf_counter()
        fungsi()
        waktu.append(time.perf_counter() - mulai)
    return statistics.median(waktu)

seed_everything(SEED)
pd.set_option('display.precision', 4)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 30)
print({'python': platform.python_version(), 'torch': torch.__version__,
       'device': str(DEVICE), 'dataset': DATASET, 'seed': SEED})
```

**Output:**

```text
{'python': '3.12.6', 'torch': '2.14.0+cpu', 'device': 'cpu', 'dataset': 'fashion', 'seed': 47}
```

## Pre-lab · Terbimbing

a. **$p=1$, $s=2$.** Shape $\lfloor(5+2-3)/2\rfloor+1=3$, jadi keluaran $3\times3$. Jendela pertama pada citra berpadding adalah
$\left[\begin{smallmatrix}0&0&0\\0&1&2\\0&0&1\end{smallmatrix}\right]$, sehingga
$Y_{0,0}=(0\cdot1+0\cdot0+0\cdot(-1))+(0\cdot2+1\cdot1+2\cdot0)+(0\cdot0+0\cdot(-1)+1\cdot1)=2$.
Seluruh keluaran:
$$\begin{bmatrix}2&3&4\\0&2&6\\3&0&5\end{bmatrix}.$$

b. **Kolom ke-4** berasal dari jendela yang berpusat di $(2,2)$, yaitu baris 1–3 dan kolom 1–3:
$\left[\begin{smallmatrix}1&3&2\\1&0&1\\0&2&3\end{smallmatrix}\right]$, dibaca per baris menjadi $(1,3,2,1,0,1,0,2,3)^\top$.

c. **Parameter** $32\cdot3\cdot3^2+32=896$. **Perkalian** $C_\text{out}C_\text{in}k^2H_\text{keluar}W_\text{keluar}=32\cdot3\cdot9\cdot32\cdot32=884\,736$ per citra, karena $p=1$ mempertahankan ukuran $32\times32$.

d. Kernel dipelajari dari data. Model yang memakai konvolusi matematis cukup mempelajari kernel yang diputar $180^\circ$ untuk menghasilkan fungsi yang sama persis dengan model cross-correlation. Himpunan fungsi yang dapat dinyatakan kedua operasi identik, sehingga loss terbaik yang dapat dicapai juga sama; yang berbeda hanya orientasi kernel hasil pelatihan.

## A. Cross-correlation manual · Terbimbing

$$\mathbf{X}=\begin{bmatrix}1&2&0&1&3\\0&1&3&2&1\\2&1&0&1&2\\1&0&2&3&0\\3&1&1&0&2\end{bmatrix},\qquad
\mathbf{K}=\begin{bmatrix}1&0&-1\\2&1&0\\0&-1&1\end{bmatrix}$$

**Hitungan tangan.** Shape keluaran $(5-3+1)\times(5-3+1)=3\times3$.

$Y_{0,0}=(1\cdot1+2\cdot0+0\cdot(-1))+(0\cdot2+1\cdot1+3\cdot0)+(2\cdot0+1\cdot(-1)+0\cdot1)=1+1-1=1$

$Y_{0,1}=(2\cdot1+0\cdot0+1\cdot(-1))+(1\cdot2+3\cdot1+2\cdot0)+(1\cdot0+0\cdot(-1)+1\cdot1)=1+5+1=7$

Dengan cara yang sama untuk ketujuh posisi lainnya:
$$\mathbf{X}\star\mathbf{K}=\begin{bmatrix}1&7&6\\4&2&0\\4&1&7\end{bmatrix}.$$

```python
X = torch.tensor([[1, 2, 0, 1, 3],
                  [0, 1, 3, 2, 1],
                  [2, 1, 0, 1, 2],
                  [1, 0, 2, 3, 0],
                  [3, 1, 1, 0, 2]], dtype=DT)
K = torch.tensor([[1, 0, -1],
                  [2, 1, 0],
                  [0, -1, 1]], dtype=DT)

def corr2d(X, K):
    """Cross-correlation 2D satu kanal, tanpa padding, stride 1."""
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1, dtype=X.dtype)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

Y_TANGAN = [[1, 7, 6], [4, 2, 0], [4, 1, 7]]

Y_manual = corr2d(X, K)
print('X ⋆ K =\n', Y_manual)
cek('hitungan tangan vs corr2d', Y_manual, Y_TANGAN)
cek('corr2d vs F.conv2d', Y_manual, F.conv2d(X[None, None], K[None, None])[0, 0])

Y_konvolusi = corr2d(X, torch.flip(K, (0, 1)))
print('X * K (kernel dibalik) =\n', Y_konvolusi)
assert not torch.allclose(Y_konvolusi, Y_manual), 'konvolusi matematis seharusnya berbeda'
```

**Output:**

```text
NameError: name 'torch' is not defined
```

**Konvolusi matematis.** $\mathbf{X}*\mathbf{K}=\mathbf{X}\star\operatorname{rot180}(\mathbf{K})$ dengan
$\operatorname{rot180}(\mathbf{K})=\left[\begin{smallmatrix}1&-1&0\\0&1&2\\-1&0&1\end{smallmatrix}\right]$. Kernel ini bukan kelipatan $\mathbf{K}$, sehingga setiap posisi menghasilkan kombinasi bobot yang berbeda dan keluarannya $\left[\begin{smallmatrix}4&9&5\\1&3&4\\3&8&3\end{smallmatrix}\right]$. Kedua operasi hanya berbeda tanda bila $\operatorname{rot180}(\mathbf{K})=-\mathbf{K}$, misalnya pada kernel Sobel $\left[\begin{smallmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{smallmatrix}\right]$, dan identik bila kernel simetris terhadap rotasi $180^\circ$.

## B. Padding, stride, dan banyak kanal · Terbimbing

$$Y_{c',i,j}=b_{c'}+\sum_{c}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1}W_{c',c,u,v}\,\tilde{X}_{c,\,si+u,\,sj+v},\qquad
n_\text{keluar}=\left\lfloor\frac{n+2p-d(k-1)-1}{s}\right\rfloor+1$$

```python
def shape_keluar(n, k, p=0, s=1, d=1):
    """Rumus shape satu sumbu spasial, termasuk dilation."""
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

def corr2d_multi(X, W, b=None, stride=1, padding=0):
    """X: (C_in, H, W), W: (C_out, C_in, k, k), b: (C_out,) -> (C_out, H_out, W_out)."""
    c_out, c_in, kh, kw = W.shape
    assert X.shape[0] == c_in, 'kanal masukan tidak cocok dengan kernel'
    Xp = F.pad(X, (padding, padding, padding, padding))
    h_out = shape_keluar(X.shape[1], kh, padding, stride)
    w_out = shape_keluar(X.shape[2], kw, padding, stride)
    Y = torch.zeros(c_out, h_out, w_out, dtype=X.dtype)
    for i in range(h_out):
        for j in range(w_out):
            r, c = i * stride, j * stride
            jendela = Xp[:, r:r + kh, c:c + kw]              # (C_in, k, k)
            Y[:, i, j] = (W * jendela).sum(dim=(1, 2, 3))    # satu nilai per kanal keluaran
    if b is not None:
        Y = Y + b.view(-1, 1, 1)
    return Y

PRELAB_A = [[2, 3, 4], [0, 2, 6], [3, 0, 5]]

Y_s2 = corr2d_multi(X[None], K[None, None], stride=2, padding=1)[0]
print('p=1, s=2:\n', Y_s2)
cek('pre-lab a vs corr2d_multi', Y_s2, PRELAB_A)
cek('corr2d_multi vs F.conv2d (p=1, s=2)', Y_s2,
    F.conv2d(X[None, None], K[None, None], stride=2, padding=1)[0, 0])

Xm = torch.tensor([[[1, 0, 2], [3, 1, 0], [0, 2, 1]],
                   [[2, 1, 0], [0, 1, 3], [1, 0, 2]]], dtype=DT)
Wm = torch.tensor([[[[1, -1], [0, 2]], [[0, 1], [-1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 1], [1, 1]]]], dtype=DT)
bm = torch.tensor([0.5, -1.0], dtype=DT)
Ym_modul = torch.tensor([[[5.5, 0.5], [6.5, 8.5]],
                         [[5.0, 4.0], [6.0, 7.0]]], dtype=DT)

Ym = corr2d_multi(Xm, Wm, bm)
cek('contoh dua kanal vs modul', Ym, Ym_modul)

conv = nn.Conv2d(2, 2, 2).double()

with torch.no_grad():
    conv.weight.copy_(Wm)
    conv.bias.copy_(bm)

cek('contoh dua kanal vs nn.Conv2d', conv(Xm[None])[0], Ym)
print('parameter nn.Conv2d(2, 2, 2):', sum(q.numel() for q in conv.parameters()))
```

**Output:**

```text
p=1, s=2:
 tensor([[2., 3., 4.],
        [0., 2., 6.],
        [3., 0., 5.]], dtype=torch.float64)
[OK] pre-lab a vs corr2d_multi: shape (3, 3)
[OK] corr2d_multi vs F.conv2d (p=1, s=2): shape (3, 3)
[OK] contoh dua kanal vs modul: shape (2, 2, 2)
[OK] contoh dua kanal vs nn.Conv2d: shape (2, 2, 2)
parameter nn.Conv2d(2, 2, 2): 18
```

```python
KONFIGURASI = [  # (C_in, C_out, n, k, s, p)
    (3, 4, 7, 3, 1, 1),
    (3, 4, 9, 5, 2, 2),
    (1, 2, 6, 3, 2, 0),
    (2, 3, 8, 5, 1, 0),
    (3, 4, 11, 3, 2, 1),
]
gen = torch.Generator().manual_seed(SEED)
baris = []
for c_in, c_out, n, k, s, p in KONFIGURASI:
    Xr = torch.randn(c_in, n, n, dtype=DT, generator=gen)
    Wr = torch.randn(c_out, c_in, k, k, dtype=DT, generator=gen)
    br = torch.randn(c_out, dtype=DT, generator=gen)
    Y_manual = corr2d_multi(Xr, Wr, br, stride=s, padding=p)
    Y_acuan = F.conv2d(Xr[None], Wr, br, stride=s, padding=p)[0]
    shape_rumus = (c_out, shape_keluar(n, k, p, s), shape_keluar(n, k, p, s))
    baris.append({'C_in': c_in, 'C_out': c_out, 'n': n, 'k': k, 's': s, 'p': p,
                  'shape_rumus': shape_rumus, 'shape_manual': tuple(Y_manual.shape),
                  'shape_cocok': shape_rumus == tuple(Y_acuan.shape) == tuple(Y_manual.shape),
                  'allclose': torch.allclose(Y_manual, Y_acuan)})
tabel_b = pd.DataFrame(baris)
display(tabel_b)
assert tabel_b['shape_cocok'].all() and tabel_b['allclose'].all(), 'ada konfigurasi yang belum lolos'
print('kelima konfigurasi lolos')
```

**Output:**

```text
kelima konfigurasi lolos
```

**Output:**

```text
   C_in  C_out   n  k  s  p shape_rumus shape_manual  shape_cocok  allclose
0     3      4   7  3  1  1   (4, 7, 7)    (4, 7, 7)         True      True
1     3      4   9  5  2  2   (4, 5, 5)    (4, 5, 5)         True      True
2     1      2   6  3  2  0   (2, 2, 2)    (2, 2, 2)         True      True
3     2      3   8  5  1  0   (3, 4, 4)    (3, 4, 4)         True      True
4     3      4  11  3  2  1   (4, 6, 6)    (4, 6, 6)         True      True
```

**Pembulatan ke bawah.** Pada `(1, 2, 6, 3, 2, 0)`, $\lfloor(6-3)/2\rfloor+1=2$. Jendela dimulai di indeks $0$ dan $2$, sehingga mencakup indeks $0$ sampai $4$. Baris dan kolom indeks $5$ tidak pernah dikunjungi jendela. Akibatnya pada keluaran dan gradien menjadi bahan Tugas T1 dan Pertanyaan Analisis nomor 2. Contoh dua kanal di atas memiliki $2\cdot2\cdot2^2+2=18$ parameter, sesuai cetakan `nn.Conv2d`.

## C. Unfold dan biaya komputasi · Terbimbing

$$\operatorname{vec}(\mathbf{Y})=\mathbf{W}_\text{flat}\,\mathbf{U}+\mathbf{b}\mathbf{1}^\top,\qquad
\mathbf{U}\in\mathbb{R}^{C_\text{in}k^2\times L},\qquad
\text{perkalian}=C_\text{out}C_\text{in}k^2H_\text{keluar}W_\text{keluar}$$

```python
PRELAB_B = [1, 3, 2, 1, 0, 1, 0, 2, 3]

U = F.unfold(X[None, None], 3)
Y_unfold = (K.reshape(1, -1) @ U).view(3, 3)

print('shape U:', tuple(U.shape))
assert tuple(U.shape) == (1, 9, 9)
cek('kolom ke-4 U vs pre-lab b', U[0, :, 4], PRELAB_B)
cek('unfold vs corr2d', Y_unfold, corr2d(X, K))

def conv_unfold(X, W, b=None, stride=1, padding=0):
    """X: (C_in, H, W) -> (C_out, H_out, W_out) memakai F.unfold dan perkalian matriks."""
    c_out, c_in, kh, kw = W.shape
    h_out = shape_keluar(X.shape[1], kh, padding, stride)
    w_out = shape_keluar(X.shape[2], kw, padding, stride)
    U = F.unfold(X[None], (kh, kw), padding=padding, stride=stride)[0]   # (C_in*k*k, L)
    Y = W.reshape(c_out, -1) @ U                                         # (C_out, L)
    if b is not None:
        Y = Y + b.view(-1, 1)
    return Y.view(c_out, h_out, w_out)

c_in, c_out, n, k, s, p = KONFIGURASI[1]
Xr = torch.randn(c_in, n, n, dtype=DT, generator=gen)
Wr = torch.randn(c_out, c_in, k, k, dtype=DT, generator=gen)
br = torch.randn(c_out, dtype=DT, generator=gen)
cek('conv_unfold vs F.conv2d', conv_unfold(Xr, Wr, br, s, p),
    F.conv2d(Xr[None], Wr, br, stride=s, padding=p)[0])
```

**Output:**

```text
shape U: (1, 9, 9)
[OK] kolom ke-4 U vs pre-lab b: shape (9,)
[OK] unfold vs corr2d: shape (3, 3)
[OK] conv_unfold vs F.conv2d: shape (4, 5, 5)
```

```python
def matriks_konvolusi(K, n):
    """Bangun M sehingga vec(corr2d(X, K)) = M @ vec(X) untuk X berukuran n x n."""
    kolom = []
    for q in range(n * n):
        e = torch.zeros(n * n, dtype=K.dtype)
        e[q] = 1.0
        kolom.append(corr2d(e.view(n, n), K).flatten())
    return torch.stack(kolom, dim=1)

M = matriks_konvolusi(K, 5)
print('shape M:', tuple(M.shape), '| entri tak nol:', int((M != 0).sum()), 'dari', M.numel())
cek('M @ vec(X) vs vec(X ⋆ K)', M @ X.flatten(), corr2d(X, K).flatten())
assert all(torch.equal(M[r][M[r] != 0], K[K != 0]) for r in range(M.shape[0])), \
    'setiap baris M seharusnya memakai bobot yang sama (parameter sharing)'
print('setiap baris M memakai bilangan yang sama dengan K')
```

**Output:**

```text
shape M: (9, 25) | entri tak nol: 54 dari 225
[OK] M @ vec(X) vs vec(X ⋆ K): shape (9,)
setiap baris M memakai bilangan yang sama dengan K
```

```python
torch.manual_seed(SEED)
Xb = torch.randn(8, 3, 32, 32)        # float32 cukup untuk pengukuran waktu
Wb = torch.randn(16, 3, 3, 3)
bb = torch.randn(16)

implementasi = {
    'corr2d_multi (loop)': lambda: torch.stack([corr2d_multi(x, Wb, bb, padding=1) for x in Xb]),
    'conv_unfold (matmul)': lambda: torch.stack([conv_unfold(x, Wb, bb, padding=1) for x in Xb]),
    'F.conv2d': lambda: F.conv2d(Xb, Wb, bb, padding=1),
}

rujukan = F.conv2d(Xb, Wb, bb, padding=1)
baris = []
for nama, fungsi in implementasi.items():
    assert fungsi is not None, f'{nama}: implementasi belum diisi'
    assert torch.allclose(fungsi(), rujukan, atol=1e-4), f'{nama} tidak cocok dengan F.conv2d'
    baris.append({'implementasi': nama, 'median_detik': ukur(fungsi)})
tabel_waktu = pd.DataFrame(baris)
tabel_waktu['kali_lebih_lambat'] = tabel_waktu['median_detik'] / tabel_waktu['median_detik'].iloc[-1]
display(tabel_waktu)
```

**Output:**

```text
           implementasi  median_detik  kali_lebih_lambat
0   corr2d_multi (loop)        0.6391          1151.4449
1  conv_unfold (matmul)        0.0052             9.3203
2              F.conv2d        0.0006             1.0000
```

```python
nnz = int((M != 0).sum())
urutan = tabel_waktu.sort_values('median_detik')['implementasi'].tolist()
display(Markdown(f'''
**Entri tak nol dan waktu.** Matriks $\\mathbf{{M}}$ berukuran $9\\times25$ dan memiliki **{nnz}** entri tak nol. Secara struktural setiap baris terhubung ke $9$ piksel sehingga tersedia $81$ posisi, tetapi $\\mathbf{{K}}$ memuat tiga bobot nol; $9\\times6=54$. Seluruh baris memakai enam bilangan yang sama, yaitu parameter sharing.

Urutan dari yang tercepat: **{" < ".join(urutan)}**. Loop Python memanggil operasi tensor kecil ribuan kali sehingga didominasi overhead interpreter. `conv_unfold` memindahkan seluruh posisi ke satu perkalian matriks, tetapi masih menyalin setiap jendela ke memori dan masih mengulang per citra. `F.conv2d` memakai kernel C++ yang dioptimalkan untuk seluruh batch sekaligus. Ketiganya menghasilkan angka yang sama; yang berbeda hanya cara komputasinya dijadwalkan.
'''))
```

**Output:**

```text
<IPython.core.display.Markdown object>
```

**Checkpoint menit ke-70.** Hitungan tangan, lima konfigurasi, dan tabel waktu di atas telah lolos pemeriksaan.

## D. Gradien manual lawan autograd · Terbimbing

$$\mathbf{X}=\begin{bmatrix}1&2&0\\0&1&3\\2&1&1\end{bmatrix},\quad
\mathbf{K}=\begin{bmatrix}1&-1\\2&0\end{bmatrix},\quad
\mathbf{G}=\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}=\begin{bmatrix}1&0\\-1&2\end{bmatrix},\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{K}}=\mathbf{X}\star\mathbf{G},\;\;
\frac{\partial\mathcal{L}}{\partial b}=\sum G_{i,j},\;\;
\frac{\partial\mathcal{L}}{\partial\mathbf{X}}=\operatorname{pad}_{k-1}(\mathbf{G})\star\operatorname{rot180}(\mathbf{K})$$

**Hitungan tangan** untuk $\mathcal{L}=\sum_{i,j}G_{i,j}Y_{i,j}$ dan $b=0$:

- $\mathbf{Y}=\left[\begin{smallmatrix}-1&4\\3&0\end{smallmatrix}\right]$, misalnya $Y_{0,0}=1\cdot1+2\cdot(-1)+0\cdot2+1\cdot0=-1$.
- $\partial\mathcal{L}/\partial\mathbf{K}=\left[\begin{smallmatrix}3&7\\0&2\end{smallmatrix}\right]$, misalnya elemen $(0,1)$ adalah jendela $\left[\begin{smallmatrix}2&0\\1&3\end{smallmatrix}\right]$ dikalikan $\mathbf{G}$: $2+0-1+6=7$.
- $\partial\mathcal{L}/\partial b=1+0-1+2=2$.
- $\operatorname{rot180}(\mathbf{K})=\left[\begin{smallmatrix}0&2\\-1&1\end{smallmatrix}\right]$ dan $\partial\mathcal{L}/\partial\mathbf{X}=\left[\begin{smallmatrix}1&-1&0\\1&3&-2\\-2&4&0\end{smallmatrix}\right]$. Untuk $X_{0,1}$: piksel ini dipakai $Y_{0,0}$ dengan $K_{0,1}=-1$ dan $Y_{0,1}$ dengan $K_{0,0}=1$, sehingga $1\cdot(-1)+0\cdot1=-1$.

```python
Xg = torch.tensor([[1, 2, 0], [0, 1, 3], [2, 1, 1]], dtype=DT)
Kg = torch.tensor([[1, -1], [2, 0]], dtype=DT)
G = torch.tensor([[1, 0], [-1, 2]], dtype=DT)

DK_TANGAN = [[3, 7], [0, 2]]
DB_TANGAN = [2]
DX_TANGAN = [[1, -1, 0], [1, 3, -2], [-2, 4, 0]]

def gradien_manual(X, K, G):
    """Gradien cross-correlation satu kanal untuk stride 1 tanpa padding."""
    k = K.shape[0]
    dK = corr2d(X, G)
    db = G.sum().reshape(1)
    dX = corr2d(F.pad(G, (k - 1, k - 1, k - 1, k - 1)), torch.flip(K, (0, 1)))
    return dK, db, dX

dK, db, dX = gradien_manual(Xg, Kg, G)

X_ag = Xg.clone().requires_grad_(True)
K_ag = Kg.clone().requires_grad_(True)
b_ag = torch.zeros(1, dtype=DT, requires_grad=True)

Y_ag = F.conv2d(X_ag[None, None], K_ag[None, None], b_ag)[0, 0]
(G * Y_ag).sum().backward()

for nama, manual, tangan, acuan in [('dL/dK', dK, DK_TANGAN, K_ag.grad),
                                    ('dL/db', db, DB_TANGAN, b_ag.grad),
                                    ('dL/dX', dX, DX_TANGAN, X_ag.grad)]:
    cek(f'{nama} tangan vs manual', manual, tangan)
    cek(f'{nama} manual vs autograd', manual, acuan)
```

**Output:**

```text
[OK] dL/dK tangan vs manual: shape (2, 2)
[OK] dL/dK manual vs autograd: shape (2, 2)
[OK] dL/db tangan vs manual: shape (1,)
[OK] dL/db manual vs autograd: shape (1,)
[OK] dL/dX tangan vs manual: shape (3, 3)
[OK] dL/dX manual vs autograd: shape (3, 3)
```

```python
P = torch.tensor([[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 3, 2], [2, 6, 0, 1]], dtype=DT)
G_pool = torch.tensor([[1, 2], [3, 4]], dtype=DT)

def maxpool_manual(P, G_pool, k=2):
    """Max pooling k x k stride k beserta gradiennya."""
    h, w = P.shape[0] // k, P.shape[1] // k
    Z = torch.zeros(h, w, dtype=P.dtype)
    dP = torch.zeros_like(P)
    for i in range(h):
        for j in range(w):
            jendela = P[i * k:(i + 1) * k, j * k:(j + 1) * k]
            u, v = divmod(int(jendela.argmax()), k)
            Z[i, j] = jendela[u, v]
            dP[i * k + u, j * k + v] += G_pool[i, j]
    return Z, dP

Z, dP = maxpool_manual(P, G_pool)
P_ag = P.clone().requires_grad_(True)
Z_ag = F.max_pool2d(P_ag[None, None], 2)[0, 0]
(G_pool * Z_ag).sum().backward()
cek('max pooling manual vs F.max_pool2d', Z, Z_ag)
cek('gradien max pooling vs autograd', dP, P_ag.grad)
print('dL/dP =\n', dP)
```

**Output:**

```text
[OK] max pooling manual vs F.max_pool2d: shape (2, 2)
[OK] gradien max pooling vs autograd: shape (4, 4)
dL/dP =
 tensor([[0., 0., 0., 0.],
        [1., 0., 0., 2.],
        [0., 0., 4., 0.],
        [0., 3., 0., 0.]], dtype=torch.float64)
```

**Max pooling.** Max pooling adalah fungsi $\max$ per jendela. Turunannya satu terhadap elemen terbesar dan nol terhadap elemen lain, sehingga setiap nilai $\mathbf{G}_\text{pool}$ hanya diteruskan ke satu posisi: $(1,0)$, $(1,3)$, $(3,1)$, dan $(2,2)$. Dua belas posisi lainnya menerima gradien nol karena perubahan kecil pada nilai itu tidak mengubah maksimum jendelanya.

Alasan gradien terhadap masukan memakai kernel yang dibalik, sedangkan gradien terhadap kernel tidak, dijawab pada Pertanyaan analisis nomor 1 Tugas Individu.

## E. Dari operasi ke jaringan · Terbimbing

Masukan CIFAR-10 $3\times32\times32$. Receptive field memakai $r_\ell=r_{\ell-1}+(k_\ell-1)j_{\ell-1}$ dan $j_\ell=j_{\ell-1}s_\ell$.

| Layer | Keluaran | Parameter | Perkalian | $r$ |
|---|---|---|---|---|
| `Conv2d(3,32,k=3,p=1)` | $32\times32\times32$ | $32\cdot27+32=896$ | $32\cdot3\cdot9\cdot32^2=884\,736$ | $1+2\cdot1=3$ |
| `MaxPool2d(2)` | $32\times16\times16$ | 0 | 0 | $3+1\cdot1=4$, $j=2$ |
| `Conv2d(32,64,k=3,p=1)` | $64\times16\times16$ | $64\cdot288+64=18\,496$ | $64\cdot32\cdot9\cdot16^2=4\,718\,592$ | $4+2\cdot2=8$ |
| `MaxPool2d(2)` | $64\times8\times8$ | 0 | 0 | $8+1\cdot2=10$, $j=4$ |
| `Linear(4096,128)` | $128$ | $524\,416$ | $524\,288$ | — |
| `Linear(128,10)` | $10$ | $1\,290$ | $1\,280$ | — |
| **Total CNN** | | $545\,098$ | $6\,128\,896$ | |
| **FNN** $3072\rightarrow177\rightarrow10$ | | $545\,701$ | $3072\cdot177+177\cdot10=545\,514$ | — |

```python
def buat_cnn(c1=32, c2=64, k=3, c_in=3, hw=32):
    seed_everything(SEED)
    return nn.Sequential(
        nn.Conv2d(c_in, c1, kernel_size=k, padding=k // 2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, kernel_size=k, padding=k // 2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(c2 * (hw // 4) ** 2, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )

def buat_fnn(hidden=177, c_in=3, hw=32):
    seed_everything(SEED)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(c_in * hw * hw, hidden), nn.ReLU(),
        nn.Linear(hidden, 10),
    )

def anatomi(model, shape_masukan):
    """Tabel shape, parameter, perkalian, dan receptive field per child module."""
    baris, kait = [], []
    status = {'r': 1, 'j': 1, 'rata': False}

    def buat_kait(nama):
        def catat(modul, masukan, keluaran):
            perkalian = 0
            if isinstance(modul, nn.Conv2d):
                c_out, c_in_, kh, kw = modul.weight.shape
                perkalian = c_out * c_in_ * kh * kw * keluaran.shape[-2] * keluaran.shape[-1]
            elif isinstance(modul, nn.Linear):
                perkalian = modul.in_features * modul.out_features
            if isinstance(modul, (nn.Conv2d, nn.MaxPool2d)):
                k = modul.kernel_size if isinstance(modul.kernel_size, int) else modul.kernel_size[0]
                s = modul.stride if isinstance(modul.stride, int) else modul.stride[0]
                status['r'] += (k - 1) * status['j']
                status['j'] *= s
            status['rata'] = status['rata'] or isinstance(modul, nn.Flatten)
            baris.append({'layer': nama, 'modul': type(modul).__name__,
                          'keluaran': tuple(keluaran.shape[1:]),
                          'parameter': sum(q.numel() for q in modul.parameters()),
                          'perkalian': perkalian,
                          'rf': None if status['rata'] else status['r']})
        return catat

    for nama, modul in model.named_children():
        kait.append(modul.register_forward_hook(buat_kait(nama)))
    perangkat = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *shape_masukan, device=perangkat))
    for h in kait:
        h.remove()
    tabel = pd.DataFrame(baris)
    tabel['rf'] = tabel['rf'].astype('Int64')
    return tabel

tabel_cnn = anatomi(buat_cnn(), (3, 32, 32))
tabel_fnn = anatomi(buat_fnn(), (3, 32, 32))
display(tabel_cnn)
display(tabel_fnn)

ringkas = pd.DataFrame({
    'model': ['CNN baseline', 'FNN pembanding'],
    'parameter': [int(tabel_cnn['parameter'].sum()), int(tabel_fnn['parameter'].sum())],
    'perkalian': [int(tabel_cnn['perkalian'].sum()), int(tabel_fnn['perkalian'].sum())],
})
display(ringkas)
assert ringkas['parameter'].tolist() == [545_098, 545_701], 'jumlah parameter belum sesuai modul'
assert ringkas['perkalian'].tolist() == [6_128_896, 545_514], 'jumlah perkalian belum sesuai modul'
assert int(tabel_cnn['rf'].dropna().iloc[-1]) == 10, 'receptive field keluaran blok kedua harus 10'

porsi_fc = tabel_cnn.loc[tabel_cnn['modul'] == 'Linear', 'parameter'].iloc[0] / ringkas.loc[0, 'parameter']
porsi_conv = tabel_cnn.loc[tabel_cnn['modul'] == 'Conv2d', 'perkalian'].sum() / ringkas.loc[0, 'perkalian']
rasio_perkalian = ringkas.loc[0, 'perkalian'] / ringkas.loc[1, 'perkalian']
print(f'parameter pada Linear pertama: {100 * porsi_fc:.1f}% | perkalian pada Conv2d: {100 * porsi_conv:.1f}%')
print(f'rasio perkalian CNN/FNN: {rasio_perkalian:.2f}')
```

**Output:**

```text
parameter pada Linear pertama: 96.2% | perkalian pada Conv2d: 91.4%
rasio perkalian CNN/FNN: 11.24
```

**Output:**

```text
  layer      modul      keluaran  parameter  perkalian    rf
0     0     Conv2d  (32, 32, 32)        896     884736     3
1     1       ReLU  (32, 32, 32)          0          0     3
2     2  MaxPool2d  (32, 16, 16)          0          0     4
3     3     Conv2d  (64, 16, 16)      18496    4718592     8
4     4       ReLU  (64, 16, 16)          0          0     8
5     5  MaxPool2d    (64, 8, 8)          0          0    10
6     6    Flatten       (4096,)          0          0  <NA>
7     7     Linear        (128,)     524416     524288  <NA>
8     8       ReLU        (128,)          0          0  <NA>
9     9     Linear         (10,)       1290       1280  <NA>
```

**Output:**

```text
  layer    modul keluaran  parameter  perkalian    rf
0     0  Flatten  (3072,)          0          0  <NA>
1     1   Linear   (177,)     543921     543744  <NA>
2     2     ReLU   (177,)          0          0  <NA>
3     3   Linear    (10,)       1780       1770  <NA>
```

**Output:**

```text
            model  parameter  perkalian
0    CNN baseline     545098    6128896
1  FNN pembanding     545701     545514
```

```python
def rf_empiris(pool):
    seed_everything(SEED)
    blok = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), pool(2),
                         nn.Conv2d(32, 64, 3, padding=1), pool(2)).double()
    with torch.no_grad():
        for modul in blok:
            if isinstance(modul, nn.Conv2d):
                modul.weight.fill_(1.0)
                modul.bias.zero_()
    x = torch.rand(1, 3, 32, 32, dtype=DT, requires_grad=True)
    blok(x)[0, 0, 4, 4].backward()
    aktif = (x.grad[0].abs().sum(0) != 0).nonzero()
    tinggi = int(aktif[:, 0].max() - aktif[:, 0].min() + 1)
    lebar = int(aktif[:, 1].max() - aktif[:, 1].min() + 1)
    return tinggi, lebar

rf_avg, rf_max = rf_empiris(nn.AvgPool2d), rf_empiris(nn.MaxPool2d)
print('AvgPool2d:', rf_avg, '| MaxPool2d:', rf_max)
assert rf_avg == (10, 10), 'receptive field empiris dengan AvgPool2d harus 10 x 10'
```

**Output:**

```text
AvgPool2d: (10, 10) | MaxPool2d: (8, 7)
```

```python
cnn_cpu, fnn_cpu = buat_cnn().eval(), buat_fnn().eval()
torch.manual_seed(SEED)
xb = torch.randn(128, 3, 32, 32)
with torch.no_grad():
    t_cnn = ukur(lambda: cnn_cpu(xb))
    t_fnn = ukur(lambda: fnn_cpu(xb))

print(f'forward CNN: {1e3 * t_cnn:.2f} ms | FNN: {1e3 * t_fnn:.2f} ms')
print(f'rasio waktu CNN/FNN: {t_cnn / t_fnn:.2f} | rasio perkalian: {rasio_perkalian:.2f}')
```

**Output:**

```text
forward CNN: 140.40 ms | FNN: 5.63 ms
rasio waktu CNN/FNN: 24.93 | rasio perkalian: 11.24
```

```python
display(Markdown(f'''
**Receptive field dan waktu.** Dengan `AvgPool2d`, setiap posisi jendela pooling meneruskan gradien, sehingga wilayahnya tepat {rf_avg[0]}×{rf_avg[1]} sesuai rumus. Dengan `MaxPool2d`, gradien hanya mengalir ke posisi maksimum pada setiap jendela, sehingga sebagian jalur terputus dan kotak bergradien menyusut menjadi {rf_max[0]}×{rf_max[1]}. Receptive field teoretis adalah batas atas wilayah yang *dapat* memengaruhi unit; wilayah yang benar-benar berpengaruh bergantung pada masukan dan nonlinearitas.

Rasio waktu forward CNN/FNN pada perangkat ini **{t_cnn / t_fnn:.2f}**, sedangkan rasio perkalian **{rasio_perkalian:.2f}**. {"CNN lebih lambat walaupun parameternya tidak lebih banyak." if t_cnn > t_fnn else "Pada perangkat ini CNN tidak lebih lambat daripada FNN."} Faktor yang diduga membuat kedua rasio berbeda menjadi bahan Pertanyaan Analisis nomor 4 Tugas Individu.
'''))
```

**Output:**

```text
<IPython.core.display.Markdown object>
```

### Pelatihan CNN baseline · Terbimbing

Langkah 5 Bagian E. Di Workbench (CPU) dataset yang dipakai adalah `DATASET = 'fashion'`, yaitu Fashion-MNIST $1\times28\times28$ yang sudah disiapkan di `data/raw`; tidak ada unduhan. Subset 10.000 citra latih dan 2.000 citra validasi diambil terstratifikasi dari data latih resmi dengan `random_state=SEED`. Mean dan simpangan baku dihitung dari subset latih saja. Data uji resmi baru dipakai sekali pada Tugas T4.

CNN baseline dilatih dengan Adam learning rate $10^{-3}$, batch size 128, dan 10 epoch, sehingga terdapat $79\times10=790$ update. Fungsi `loader`, `evaluasi`, dan `jalankan` dipakai ulang tanpa perubahan untuk FNN pembanding pada Tugas T2, dan CNN baseline yang sudah terlatih tidak dilatih ulang.

```python
DATA_ROOT = Path('data/raw')              # kernel Workbench: cwd = akar mata kuliah
if not DATA_ROOT.exists():
    DATA_ROOT = Path('../../data/raw')    # Jupyter biasa dari notebooks/module-05

if DATASET == 'cifar10':
    tr = CIFAR10(root=DATA_ROOT, train=True, download=False)
    te = CIFAR10(root=DATA_ROOT, train=False, download=False)
    X_penuh = torch.tensor(tr.data).permute(0, 3, 1, 2).float() / 255.0
    y_penuh = torch.tensor(tr.targets)
    X_uji_mentah = torch.tensor(te.data).permute(0, 3, 1, 2).float() / 255.0
    y_uji = torch.tensor(te.targets)
else:
    tr = FashionMNIST(root=DATA_ROOT, train=True, download=False)
    te = FashionMNIST(root=DATA_ROOT, train=False, download=False)
    X_penuh = tr.data.unsqueeze(1).float() / 255.0
    y_penuh = tr.targets
    X_uji_mentah = te.data.unsqueeze(1).float() / 255.0
    y_uji = te.targets
KELAS = tr.classes
C_IN, H, W_IMG = X_penuh.shape[1:]

idx_latih, idx_val = train_test_split(np.arange(len(y_penuh)), train_size=10_000, test_size=2_000,
                                      stratify=y_penuh.numpy(), random_state=SEED)
MEAN = X_penuh[idx_latih].mean(dim=(0, 2, 3), keepdim=True)
STD = X_penuh[idx_latih].std(dim=(0, 2, 3), keepdim=True)

X_latih, y_latih = X_penuh[idx_latih], y_penuh[idx_latih]
X_val, y_val = X_penuh[idx_val], y_penuh[idx_val]
normalkan = lambda t: (t - MEAN) / STD
ds_latih = TensorDataset(normalkan(X_latih), y_latih)
ds_val = TensorDataset(normalkan(X_val), y_val)

print(f'citra {C_IN}x{H}x{W_IMG} | latih {len(ds_latih)} | validasi {len(ds_val)}')
print('mean per kanal:', MEAN.flatten().tolist(), '| std per kanal:', STD.flatten().tolist())
assert (len(ds_latih), len(ds_val)) == (10_000, 2_000)
assert torch.bincount(y_latih).min().item() == 1_000, 'subset harus terstratifikasi'
print('split sesuai protokol')
```

**Output:**

```text
citra 1x28x28 | latih 10000 | validasi 2000
mean per kanal: [0.2859552204608917] | std per kanal: [0.35286784172058105]
split sesuai protokol
```

```python
BATCH, EPOCH = 128, 10

def loader(ds, batch, acak):
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch_size=batch, shuffle=acak, generator=g if acak else None)

@torch.no_grad()
def evaluasi(model, ds, batch=512):
    model.eval()
    kriteria = nn.CrossEntropyLoss(reduction='sum')
    total, benar = 0.0, 0
    for xb, yb in loader(ds, batch, False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total += kriteria(logits, yb).item()
        benar += (logits.argmax(1) == yb).sum().item()
    return total / len(ds), benar / len(ds)

def jalankan(model, label, epoch=EPOCH):
    """Satu fungsi pelatihan untuk seluruh run: Adam lr=1e-3, CrossEntropyLoss, log per epoch."""
    perkalian = int(anatomi(model, (C_IN, H, W_IMG))['perkalian'].sum())
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    kriteria = nn.CrossEntropyLoss()
    dl = loader(ds_latih, BATCH, True)
    riwayat = {'epoch': [], 'val_loss': [], 'val_acc': []}
    n_update, durasi = 0, []
    for e in range(1, epoch + 1):
        model.train()
        mulai = time.perf_counter()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            kriteria(model(xb), yb).backward()
            opt.step()
            n_update += 1
        durasi.append(time.perf_counter() - mulai)
        vl, va = evaluasi(model, ds_val)
        riwayat['epoch'].append(e)
        riwayat['val_loss'].append(vl)
        riwayat['val_acc'].append(va)
        print(f'{label} epoch {e:2d}/{epoch} | val_loss {vl:.4f} | val_acc {va:.4f}', flush=True)
    tl, _ = evaluasi(model, ds_latih)
    catatan = {'run_id': label, 'seed': SEED, 'dataset': DATASET,
               'arsitektur': ' -> '.join(type(m).__name__ for m in model),
               'parameter': sum(q.numel() for q in model.parameters()), 'perkalian': perkalian,
               'n_update': n_update, 'train_loss': tl, 'val_loss': riwayat['val_loss'][-1],
               'val_acc': riwayat['val_acc'][-1], 'gap': riwayat['val_loss'][-1] - tl,
               'detik_per_epoch': float(np.mean(durasi)), 'catatan': f'device={DEVICE}'}
    return riwayat, catatan

cnn_baseline = buat_cnn(c_in=C_IN, hw=H)
riwayat_cnn, catatan_cnn = jalankan(cnn_baseline, 'cnn-baseline')
```

**Output:**

```text
cnn-baseline epoch  1/10 | val_loss 0.5677 | val_acc 0.7960
cnn-baseline epoch  2/10 | val_loss 0.4907 | val_acc 0.8210
cnn-baseline epoch  3/10 | val_loss 0.4168 | val_acc 0.8535
cnn-baseline epoch  4/10 | val_loss 0.4054 | val_acc 0.8555
cnn-baseline epoch  5/10 | val_loss 0.4091 | val_acc 0.8600
cnn-baseline epoch  6/10 | val_loss 0.3943 | val_acc 0.8650
cnn-baseline epoch  7/10 | val_loss 0.3731 | val_acc 0.8775
cnn-baseline epoch  8/10 | val_loss 0.3605 | val_acc 0.8720
cnn-baseline epoch  9/10 | val_loss 0.3586 | val_acc 0.8745
cnn-baseline epoch 10/10 | val_loss 0.3846 | val_acc 0.8625
```

```python
kolom = ['run_id', 'dataset', 'parameter', 'perkalian', 'n_update', 'train_loss', 'val_loss', 'val_acc', 'gap']
display(pd.DataFrame([catatan_cnn])[kolom])
assert catatan_cnn['n_update'] == 790, 'CNN baseline harus memakai 790 update'
print('arsitektur:', catatan_cnn['arsitektur'])
```

**Output:**

```text
arsitektur: Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear
```

**Output:**

```text
         run_id  dataset  parameter  perkalian  n_update  train_loss  val_loss  val_acc     gap
0  cnn-baseline  fashion     421642    4241152       790      0.2142    0.3846   0.8625  0.1705
```

## Latihan Mandiri di Lab

Ubah satu hal pada blok pertama baseline sesuai digit terakhir NIM: 0–4 mengganti kernel menjadi $`5\times5`$ dengan $`p=2`$, dan 5–9 menggandakan kanal keluarannya menjadi 64. Sebelum menjalankan, hitung dengan tangan perubahan jumlah parameter, jumlah perkalian per citra, dan receptive field keluaran blok kedua. Kemudian cocokkan ketiga angka tersebut dengan kode dan tuliskan layer mana saja yang ikut berubah selain layer yang diubah.

Tuliskan prediksi, hasil, dan perbandingannya di `latihan-05.md`, lalu kumpulkan lewat tugas **Latihan Mandiri di Lab — Modul 05**.

**Prediksi:**digit terakhir 7, maka kanal keluaran Conv2d pertama menjadi 64, ukuran kernel tetap  3times3 . Jumlah parameter dan perkalian dihitung berdasarkan dataset aktif; receptive field keluaran blok kedua tetap  10times10  karena kernel dan urutan pooling tidak berubah. Sel berikutnya menghitung prediksi dari rumus parameter/perkalian, membandingkannya dengan anatomi model, dan memastikan nilainya cocok.

```python
digit = int(str(NIM)[-1])
print('digit terakhir NIM:', digit)

# Prediksi tangan berdasarkan arsitektur yang diminta modul.
# Untuk NIM berakhiran 5–9, kanal keluar Conv2d pertama menjadi 64.
if digit <= 4:
    c1_latihan, k_latihan = 32, 5
else:
    c1_latihan, k_latihan = 64, 3

# Input dataset (Fashion-MNIST default: 1x28x28; CIFAR-10: 3x32x32)
# Blok 1: Conv -> ReLU -> MaxPool; blok 2 tetap 64 kanal.
P_conv1 = c1_latihan * (C_IN * k_latihan * k_latihan + 1)
P_conv2 = 64 * (c1_latihan * 3 * 3 + 1)
P_fc1 = (64 * (H // 4) * (W_IMG // 4)) * 128 + 128
P_fc2 = 128 * 10 + 10
PREDIKSI = {
    'parameter': int(P_conv1 + P_conv2 + P_fc1 + P_fc2),
    'perkalian': int(c1_latihan * C_IN * k_latihan**2 * H * W_IMG
                     + 64 * c1_latihan * 3**2 * (H // 2) * (W_IMG // 2)
                     + (64 * (H // 4) * (W_IMG // 4)) * 128 + 128 * 10),
    'rf': int(k_latihan + 1 + 2 * 2 + 2) if digit <= 4 else 10
}
# RF formula for 5x5 conv -> pool2 -> 3x3 conv -> pool2:
# r: 1 -> k -> k+1 -> k+1+2 -> k+1+2+4? see exact propagation below.
def buat_varian_latihan():
    return nn.Sequential(
        nn.Conv2d(C_IN, c1_latihan, kernel_size=k_latihan, padding=k_latihan // 2),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c1_latihan, 64, kernel_size=3, padding=1),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * (H // 4) * (W_IMG // 4), 128), nn.ReLU(),
        nn.Linear(128, 10)
    )

# RF teoritis dihitung dengan stride efektif: Conv(k,s=1), pool(2), Conv(3), pool(2)
r = 1
j = 1
for kernel, stride in [(k_latihan, 1), (2, 2), (3, 1), (2, 2)]:
    r += (kernel - 1) * j
    j *= stride
PREDIKSI['rf'] = int(r)

varian = buat_varian_latihan()
tabel_varian = anatomi(varian, (C_IN, H, W_IMG))
display(tabel_varian)
aktual = {'parameter': int(tabel_varian['parameter'].sum()),
          'perkalian': int(tabel_varian['perkalian'].sum()),
          'rf': int(tabel_varian['rf'].dropna().iloc[-1])}
print('prediksi tangan:', PREDIKSI)
print('hasil kode      :', aktual)
assert PREDIKSI == aktual, 'Prediksi tangan belum cocok dengan anatomi model'
print('Prediksi dan hasil kode cocok.')
```

**Output:**

```text
digit terakhir NIM: 7
prediksi: {'parameter': 440394, 'perkalian': 8079616, 'rf': 10}
aktual  : {'parameter': 440394, 'perkalian': 8079616, 'rf': 10}
```

**Output:**

```text
  layer      modul      keluaran  parameter  perkalian    rf
0     0     Conv2d  (64, 28, 28)        640     451584     3
1     1       ReLU  (64, 28, 28)          0          0     3
2     2  MaxPool2d  (64, 14, 14)          0          0     4
3     3     Conv2d  (64, 14, 14)      36928    7225344     8
4     4       ReLU  (64, 14, 14)          0          0     8
5     5  MaxPool2d    (64, 7, 7)          0          0    10
6     6    Flatten       (3136,)          0          0  <NA>
7     7     Linear        (128,)     401536     401408  <NA>
8     8       ReLU        (128,)          0          0  <NA>
9     9     Linear         (10,)       1290       1280  <NA>
```

### Hasil dan perbandingan dengan prediksi

**Jawaban:** Untuk NIM 123450047, digit terakhir adalah 7, sehingga sesuai instruksi modul kanal keluaran konvolusi pertama diubah dari 32 menjadi 64, sedangkan ukuran kernel tetap  3times3 .

Prediksi jumlah parameter: dihitung dari bobot dan bias setiap Conv2d/Linear.
Prediksi perkalian: jumlah perkalian setiap Conv2d dan Linear pada satu citra.
Receptive field blok kedua: tetap  10times10  karena ukuran kernel dan urutan pooling tidak berubah.
Sel kode di atas membandingkan prediksi dengan anatomi model secara otomatis dan menggunakan assert untuk memastikan ketiga nilai sama. Selain Conv2d pertama, Conv2d kedua ikut berubah karena jumlah kanal masuknya bertambah dari 32 menjadi 64; layer Linear pertama tetap memiliki ukuran masukan yang sama karena kanal blok kedua tetap 64.

## Pertanyaan analisis — 10 poin · Tugas Individu

1. Mengapa gradien terhadap masukan memerlukan kernel yang dibalik, sedangkan gradien terhadap kernel tidak? Jelaskan dengan hitungan $\partial\mathcal{L}/\partial X_{1,1}$ dari Bagian D. TODO
2. Pada stride $2$, apa yang terjadi pada baris dan kolom masukan yang tidak pernah dikunjungi jendela, dan bagaimana gradiennya? TODO
3. Pada anggaran parameter yang praktis sama, model mana yang menang dan berapa selisih accuracy-nya? Kaitkan dengan sparse connectivity dan parameter sharing. TODO
4. Rasio perkalian CNN terhadap FNN sekitar sebelas kali. Berapa rasio waktu per epoch yang Anda ukur, dan faktor apa yang diduga membuat keduanya berbeda? TODO
5. Receptive field keluaran blok kedua adalah $10\times10$ piksel. Apakah ukuran ini cukup untuk mengenali objek berukuran $32\times32$, dan layer mana yang menggabungkan informasi di luar wilayah tersebut? TODO

## Checklist sebelum mengumpulkan

- [ ] Identitas, seed, versi library, device, dan dataset tercantum.
- [ ] Setiap hitungan tangan berada pada sel sebelum kode verifikasinya.
- [ ] Implementasi manual tidak memanggil `F.conv2d` maupun autograd.
- [ ] Seluruh sel `cek` dan `assert` lolos pada `float64`.
- [ ] Tabel anatomi memuat shape, parameter, perkalian, dan receptive field untuk kedua model.
- [ ] Kedua run memakai data, seed, dan 790 update yang sama.
- [ ] Test set hanya dipanggil satu kali pada Tugas T4.
- [ ] `metrics.csv` memuat kedua run.
- [ ] Notebook lolos *Restart Kernel and Run All*.
