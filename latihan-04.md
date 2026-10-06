Latihan Mandiri di Lab
Bandingkan dropout 0,2 dan 0,5 pada konfigurasi baseline. Sebelum menjalankan, tuliskan prediksi Anda tentang arah perubahan train loss, validation loss, dan gap. Setelah itu jelaskan mengapa dropout yang terlalu besar dapat memperburuk kedua angka sekaligus.

Tuliskan prediksi, hasil, dan perbandingannya di latihan-04.md, lalu kumpulkan lewat tugas Latihan Mandiri di Lab — Modul 04.

Prediksi: Dropout 0.2 diperkirakan menurunkan overfitting dibanding baseline sehingga gap mengecil dan validation loss membaik. Dropout 0.5 diperkirakan memberi regularisasi lebih kuat sehingga train loss lebih tinggi. Jika terlalu besar, dropout dapat menyebabkan underfitting sehingga train loss dan validation loss sama-sama memburuk.

Code :
# Latihan Mandiri di Lab
_, history_drop02, note_drop02 = jalankan(
    hidden=512,
    dropout=0.2,
    weight_decay=0.0,
    lr=1e-3,
    epoch=EPOCH,
    sabar=None,
    label='dropout_0.2'
)

hasil_banding = pd.DataFrame([
    catatan_baseline,
    df_awal[df_awal['run_id'] == 'dropout_0.5'].iloc[0].to_dict(),
    note_drop02
])[['run_id','train_loss','val_loss','gap','val_acc']]

display(hasil_banding)
print('Bandingkan nilai train_loss, val_loss, dan gap pada tabel di atas.')

Output:
Bandingkan nilai train_loss, val_loss, dan gap pada tabel di atas.




run_id

train_loss

val_loss

gap

val_acc

0

baseline

0.2036

0.4646

0.2610

0.8453

1

dropout_0.5

0.2453

0.4526

0.2073

0.8410

2

dropout_0.2

0.2463

0.4509

0.2046

0.8437

Hasil dan perbandingan dengan prediksi
Jawaban: Berdasarkan hasil percobaan, penggunaan dropout berhasil mengurangi overfitting dibandingkan model baseline. Hal ini terlihat dari nilai **gap** antara train loss dan validation loss yang menurun dari **0,2610** pada baseline menjadi **0,2073** pada dropout 0,5 dan **0,2046** pada dropout 0,2.

Nilai **train_loss** pada baseline adalah **0,2036**, sedangkan pada dropout 0,5 dan dropout 0,2 meningkat menjadi **0,2453** dan **0,2463**. Kenaikan train loss ini wajar karena dropout menonaktifkan sebagian neuron selama proses pelatihan sehingga model lebih sulit menghafal data latih.

Untuk **val_loss**, kedua model dengan dropout memberikan hasil yang lebih baik dibanding baseline. Baseline memiliki val_loss **0,4646**, sedangkan dropout 0,5 menghasilkan **0,4526** dan dropout 0,2 menghasilkan **0,4509**, yang merupakan nilai terbaik di antara ketiganya.

Pada **val_acc**, baseline memperoleh **0,8453**, dropout 0,2 memperoleh **0,8437**, dan dropout 0,5 memperoleh **0,8410**. Meskipun akurasi validasi sedikit lebih rendah dibanding baseline, kedua model dengan dropout menunjukkan generalisasi yang lebih baik karena memiliki gap yang lebih kecil dan validation loss yang lebih rendah.

Hasil ini sesuai dengan prediksi awal bahwa dropout dapat mengurangi overfitting. Selain itu, dropout 0,2 memberikan keseimbangan terbaik antara regularisasi dan kemampuan belajar model karena menghasilkan validation loss terendah dan gap terkecil dibanding konfigurasi lainnya.
