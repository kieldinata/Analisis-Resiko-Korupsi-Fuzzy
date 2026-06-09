import streamlit as st
import pandas as pd
import plotly.express as px
import machine_learning as ml
import numpy as np
import math
import joblib

config = {
    'Metode Pengadaan': {
        'type': 'map',
        'data': {'E-Purchasing': 0.3, 'Pengadaan Langsung': 0.6, 'Dikecualikan': 1.0}
    },
    'Jenis Pengadaan': {
        'type': 'map',
        'data': {'Barang': 0.3, 'Jasa Lainnya': 0.6, 'Jasa Konsultansi': 1.0}
    },
    'Cara Pengadaan': {
        'type': 'map',
        'data': {'Penyedia': 0.4, 'Swakelola': 0.8}
    },
    'Sumber Dana': {
        'type': 'map',
        'data': {'APBD': 0.3, 'APBDP': 0.6, 'APBD; APBDP': 1.0}
    },
    'Total Nilai (Rp)': {
        'type': 'threshold',
        'limit': 5000000000
    }
}

def tampilkan_visualisasi_3d_ml():
    try:
        model = joblib.load('model_urgensi.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        fitur = vectorizer.get_feature_names_out()
        bobot = model.coef_
        
        if len(fitur) == 0:
            st.warning("Model belum mempelajari kata kunci apa pun.")
            return

        df_otak_ml = pd.DataFrame({
            'Kata_Kunci': fitur,
            'Indeks_Kata': np.arange(len(fitur)),
            'Bobot_Koefisien': bobot,
            'Nilai_Mutlak_Pengaruh': np.abs(bobot)
        })
       
        df_otak_ml = df_otak_ml.sort_values(by='Bobot_Koefisien', ascending=False).reset_index(drop=True)
        df_otak_ml['Urutan_Bobot'] = df_otak_ml.index

        fig = px.scatter_3d(
            df_otak_ml, 
            x='Urutan_Bobot',
            y='Bobot_Koefisien',
            z='Nilai_Mutlak_Pengaruh',
            text='Kata_Kunci',
            color='Bobot_Koefisien',
            color_continuous_scale='RdYlBu',
            size='Nilai_Mutlak_Pengaruh',
            size_max=18,
            labels={
                'Urutan_Bobot': 'Struktur Kosakata Model',
                'Bobot_Koefisien': 'Nilai Koefisien (Beta)',
                'Nilai_Mutlak_Pengaruh': 'Skala Sensitivitas Kata'
            }
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("File 'model_urgensi.pkl' atau 'vectorizer.pkl' tidak ditemukan.")


def aplikasikan_basis_aturan(row, ml_score):
    """
    1. Parameter:
       - row (pandas.Series): Baris data paket pengadaan yang sedang diproses.
       - ml_score (float): Skor prediksi tingkat urgensi teks dari model Machine Learning (0.0 - 1.0).
    2. Overview:
       - Mengevaluasi kombinasi teks nama paket, metode pengadaan, dan sumber dana berdasarkan aturan paksa (expert rules).
       - Menghitung faktor pengali bobot secara dinamis menyesuaikan nominal angka total proyek (makin besar nominal, penalti makin galak).
    3. Output:
       - float: Nilai akumulasi penalti risiko (skor bonus) yang akan ditambahkan secara brutal ke indeks risiko fuzzy.
    """
    penalti = 0.0
    nama_paket = str(row['Nama Paket']).lower()
    metode = str(row['Metode Pengadaan'])
    cara_pengadaan = str(row['Cara Pengadaan'])
    sumber_dana = str(row['Sumber Dana'])
    nilai_aktual = float(row['Total Nilai (Rp)'])
    
    if nilai_aktual < 200000000:
        pengali_nilai = 0.5
    elif nilai_aktual < 1000000000:
        pengali_nilai = 0.8
    elif nilai_aktual < 5000000000:
        pengali_nilai = 1.0
    elif nilai_aktual < 20000000000:
        pengali_nilai = 1.3
    else:
        pengali_nilai = 1.6

    if 'vip' in nama_paket or 'vvip' in nama_paket:
        penalti += 0.30 * pengali_nilai
    if 'kendaraan dinas' in nama_paket or 'mobil dinas' in nama_paket:
        penalti += 0.20 * pengali_nilai
    if 'rumah dinas' in nama_paket or 'rumah jabatan' in nama_paket:
        penalti += 0.15 * pengali_nilai

    if 'perjalanan dinas' in nama_paket or 'perlesiran' in nama_paket:
        penalti += 0.25 * pengali_nilai
    if 'pakaian dinas' in nama_paket or 'seragam' in nama_paket:
        penalti += 0.15 * pengali_nilai
    if 'studi banding' in nama_paket or 'kunjungan kerja' in nama_paket:
        penalti += 0.20 * pengali_nilai
    if 'pameran' in nama_paket or 'event organizer' in nama_paket:
        penalti += 0.15 * pengali_nilai

    if ('vip' in nama_paket) and (metode == 'Dikecualikan'):
        penalti += 0.15 * pengali_nilai
    if ('perjalanan dinas' in nama_paket) and (metode == 'Dikecualikan'):
        penalti += 0.15 * pengali_nilai
    if ('pakaian' in nama_paket or 'seragam' in nama_paket) and (metode == 'Pengadaan Langsung') and (nilai_aktual > 180000000):
        penalti += 0.20

    if ('perjalanan dinas' in nama_paket or 'studi banding' in nama_paket) and (cara_pengadaan == 'Swakelola'):
        penalti += 0.20 * pengali_nilai
    if (cara_pengadaan == 'Swakelola') and (nilai_aktual > 2000000000):
        penalti += 0.25 * pengali_nilai

    if (ml_score > 0.8) and ('penunjang' in nama_paket or 'pendukung' in nama_paket):
        penalti += 0.15 * pengali_nilai
    if (nilai_aktual > 50000000000) and (ml_score > 0.85):
        penalti += 0.30
    if ('pembangunan gedung' in nama_paket or 'konstruksi' in nama_paket) and (metode == 'Dikecualikan'):
        penalti += 0.20 * pengali_nilai

    if ('pembelian' in nama_paket or 'pengadaan' in nama_paket) and (sumber_dana == 'APBDP'):
        penalti += 0.10 * pengali_nilai
    if ('sosialisasi' in nama_paket or 'bimtek' in nama_paket) and (nilai_aktual > 1000000000):
        penalti += 0.25

    return penalti

def sugeno_score(row, ml_score):
    """
    1. Parameter:
       - row (pandas.Series): Baris data paket pengadaan berisi skor parameter terpetakan.
       - ml_score (float): Skor prediksi tingkat urgensi dari model Machine Learning.
    2. Overview:
       - Fuzzifikasi murni dari nol untuk 5 parameter administratif, diuji dengan 15 aturan, didesain memenuhi rubrik.
       - Penalti expert digabungkan di akhir sebagai komponen hibrida.
    3. Output:
       - float: Nilai akhir indeks risiko Sugeno maksimal 1.0.
    """
    m = float(row['Metode Pengadaan_Score'])
    j = float(row['Jenis Pengadaan_Score'])
    s = float(row['Sumber Dana_Score'])
    n = float(row['Total Nilai (Rp)_Score'])
    c = float(row['Cara Pengadaan_Score'])
    u = 1.0 - float(ml_score)

    m_R, m_S, m_T = max(0.0, (0.5 - m)/0.5), max(0.0, (0.5 - abs(m - 0.5))/0.5), max(0.0, (m - 0.5)/0.5)
    j_R, j_S, j_T = max(0.0, (0.5 - j)/0.5), max(0.0, (0.5 - abs(j - 0.5))/0.5), max(0.0, (j - 0.5)/0.5)
    s_R, s_S, s_T = max(0.0, (0.5 - s)/0.5), max(0.0, (0.5 - abs(s - 0.5))/0.5), max(0.0, (s - 0.5)/0.5)
    n_R, n_S, n_T = max(0.0, (0.5 - n)/0.5), max(0.0, (0.5 - abs(n - 0.5))/0.5), max(0.0, (n - 0.5)/0.5)
    c_R, c_S, c_T = max(0.0, (0.5 - c)/0.5), max(0.0, (0.5 - abs(c - 0.5))/0.5), max(0.0, (c - 0.5)/0.5)
    u_R, u_S, u_T = max(0.0, (0.5 - u)/0.5), max(0.0, (0.5 - abs(u - 0.5))/0.5), max(0.0, (u - 0.5)/0.5)

    a1 = min(n_T, u_T)      # Nominal tinggi dan urgensi ML rendah
    a2 = min(m_T, n_T)      # Metode berisiko dan nominal tinggi
    a3 = min(m_T, u_T)      # Metode berisiko dan urgensi ML rendah
    a4 = min(j_T, n_T)      # Jenis berisiko dan nominal tinggi
    a5 = min(s_T, n_T)      # Sumber dana berisiko dan nominal tinggi
    a14 = min(n_T, c_T)     # Nominal tinggi dan cara swakelola
    a15 = min(u_T, c_T)     # Urgensi ML rendah dan cara swakelola

    a6 = min(n_S, u_T)      # Nominal sedang dan urgensi ML rendah
    a7 = min(m_S, n_S)      # Metode sedang dan nominal sedang
    a8 = min(j_S, u_S)      # Jenis sedang dan urgensi ML sedang
    a9 = min(s_S, u_S)      # Sumber dana sedang dan urgensi ML sedang
    a16 = min(c_S, n_S)     # Cara sedang dan nominal sedang
    a17 = min(c_S, u_S)     # Cara sedang dan urgensi ML sedang

    a10 = min(m_R, u_R)     # Metode aman dan urgensi ML tinggi
    a11 = min(n_R, u_R)     # Nominal rendah dan urgensi ML tinggi
    a12 = min(j_R, n_R)     # Jenis aman dan nominal rendah
    a13 = min(s_R, m_R)     # Sumber dana aman dan metode aman
    a18 = min(c_R, m_R)     # Cara penyedia dan metode aman
    a19 = min(c_R, n_R)     # Cara penyedia dan nominal rendah
    a20 = min(c_R, u_R)     # Cara penyedia dan urgensi ML tinggi
    a21 = min(c_R, j_R)     # Cara penyedia dan jenis aman
    
    num = (
        (a1 * 0.9) + (a2 * 0.8) + (a3 * 0.8) + (a4 * 0.7) + (a5 * 0.7) + (a14 * 0.85) + (a15 * 0.75) +
        (a6 * 0.6) + (a7 * 0.5) + (a8 * 0.5) + (a9 * 0.4) + (a16 * 0.5) + (a17 * 0.5) + (a10 * 0.2) +
        (a11 * 0.1) + (a12 * 0.1) + (a13 * 0.1) + (a18 * 0.1) + (a19 * 0.1) + (a20 * 0.15) + (a21 * 0.1)
    )
    
    den = sum([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21])
    
    score_fuzzy = num / (den + 1e-9)
    penalti_expert = aplikasikan_basis_aturan(row, ml_score)
    
    return min(score_fuzzy + penalti_expert, 1.0)

def mamdani_score(row, ml_score):
    """
    1. Parameter:
       - row (pandas.Series): Baris data paket pengadaan berisi skor parameter terpetakan.
       - ml_score (float): Skor prediksi tingkat urgensi dari model Machine Learning.
    2. Overview:
       - Fuzzifikasi, 15 evaluasi aturan Mamdani, dan defuzzifikasi (Centroid) untuk nilai administrasi.
       - Penalti teks/nominal ekstrem diaplikasikan sebagai override di akhir proses.
    3. Output:
       - float: Nilai akhir indeks risiko Mamdani maksimal 1.0.
    """
    m = float(row['Metode Pengadaan_Score'])
    j = float(row['Jenis Pengadaan_Score'])
    s = float(row['Sumber Dana_Score'])
    n = float(row['Total Nilai (Rp)_Score'])
    c = float(row['Cara Pengadaan_Score'])
    u = 1.0 - float(ml_score)
    
    m_R, m_S, m_T = max(0.0, (0.5 - m)/0.5), max(0.0, (0.5 - abs(m - 0.5))/0.5), max(0.0, (m - 0.5)/0.5)
    j_R, j_S, j_T = max(0.0, (0.5 - j)/0.5), max(0.0, (0.5 - abs(j - 0.5))/0.5), max(0.0, (j - 0.5)/0.5)
    s_R, s_S, s_T = max(0.0, (0.5 - s)/0.5), max(0.0, (0.5 - abs(s - 0.5))/0.5), max(0.0, (s - 0.5)/0.5)
    n_R, n_S, n_T = max(0.0, (0.5 - n)/0.5), max(0.0, (0.5 - abs(n - 0.5))/0.5), max(0.0, (n - 0.5)/0.5)
    c_R, c_S, c_T = max(0.0, (0.5 - c)/0.5), max(0.0, (0.5 - abs(c - 0.5))/0.5), max(0.0, (c - 0.5)/0.5)
    u_R, u_S, u_T = max(0.0, (0.5 - u)/0.5), max(0.0, (0.5 - abs(u - 0.5))/0.5), max(0.0, (u - 0.5)/0.5)
    
    a1, a2, a3 = min(n_T, u_T), min(m_T, n_T), min(m_T, u_T)
    a4, a5 = min(j_T, n_T), min(s_T, n_T)
    a6, a7, a8, a9 = min(n_S, u_T), min(m_S, n_S), min(j_S, u_S), min(s_S, u_S)
    a10, a11, a12, a13 = min(m_R, u_R), min(n_R, u_R), min(j_R, n_R), min(s_R, m_R)
    a14, a15 = min(n_T, c_T), min(u_T, c_T)                                                 # Nominal tinggi dan cara swakelola | Urgensi ML rendah dan cara swakelola
    a16, a17 = min(c_S, n_S), min(c_S, u_S)                                                 # Cara sedang dan nominal sedang | Cara sedang dan urgensi ML sedang
    a18, a19, a20, a21 = min(c_R, m_R), min(c_R, n_R), min(c_R, u_R), min(c_R, j_R)         # Aturan Cara Penyedia dengan parameter aman lainnya
    
    mu_SR = max(a11, a12, a13, a18, a19, a21)  # Output: Sangat Rendah (ditambah cara penyedia yang aman)
    mu_R = max(a10, a20)                       # Output: Rendah (ditambah cara penyedia dan urgensi ML tinggi)
    mu_S = max(a7, a8, a9, a16, a17)           # Output: Sedang (ditambah kondisi cara pengadaan sedang)
    mu_T = max(a4, a5, a6, a15)                # Output: Tinggi (tetap menampung swakelola dengan urgensi rendah)
    mu_ST = max(a1, a2, a3, a14)               # Output: Sangat Tinggi (tetap menampung swakelola nominal tinggi)
    
    num, den = 0.0, 0.0
    for step in range(11):
        k = step / 10.0
        f_SR = max(0.0, (0.2 - abs(k - 0.1)) / 0.2)
        f_R = max(0.0, (0.2 - abs(k - 0.3)) / 0.2)
        f_S = max(0.0, (0.2 - abs(k - 0.5)) / 0.2)
        f_T = max(0.0, (0.2 - abs(k - 0.7)) / 0.2)
        f_ST = max(0.0, (0.2 - abs(k - 0.9)) / 0.2)
        mu_k = max(min(f_SR, mu_SR), min(f_R, mu_R), min(f_S, mu_S), min(f_T, mu_T), min(f_ST, mu_ST))
        num += k * mu_k
        den += mu_k
        
    score_fuzzy = num / (den + 1e-9)
    penalti_expert = aplikasikan_basis_aturan(row, ml_score)
    
    return min(score_fuzzy + penalti_expert, 1.0)

def klasifikasi_risiko(skor_0_sampai_1):
    """
    1. Parameter:
       - skor_0_sampai_1 (float): Nilai indeks risiko gabungan/tunggal dalam skala desimal.
    2. Overview:
       - Mengonversi nilai skor desimal ke dalam persentase skala 100.
       - Melakukan klasifikasi pengondisian berdasarkan ambang batas untuk menentukan label kategori kerawanan anggaran daerah.
    3. Output:
       - str: Teks label kategori risiko ("Sangat Tidak Rawan", "Sedikit Rawan", "Cukup Rawan", "Sangat Rawan").
    """
    skor_100 = skor_0_sampai_1 * 100
    if skor_100 < 10: return "Sangat Tidak Rawan"
    if skor_100 < 25: return "Sedikit Rawan"
    if skor_100 < 45: return "Cukup Rawan"
    return "Sangat Rawan"

@st.cache_data
def process_data_single_file(file_content, config_dict):
    """
    1. Parameter:
       - file_content (BytesIO / UploadedFile): Berkas CSV mentah yang diunggah oleh pengguna.
       - config_dict (dict): Kamus konfigurasi pemetaan skor logika fuzzy.
    2. Overview:
       - Membaca file CSV pengadaan menggunakan pandas, lalu melakukan iterasi kolom.
       - Mentransformasikan nilai kategori administrasi ke bentuk skor numerik fuzzy (mapping) dan mengubah nilai nominal riil uang menggunakan transformasi skala logaritmik berbasis 10.
    3. Output:
       - pandas.DataFrame: DataFrame baru yang sudah diperkaya dengan kolom skor numerik terfuzzifikasi (*_Score*).
    """
    df = pd.read_csv(file_content)
    for col, cfg in config_dict.items():
        if col in df.columns:
            if cfg['type'] == 'map':
                df[f'{col}_Score'] = df[col].map(cfg['data']).fillna(0)
            elif cfg['type'] == 'threshold':
                df[f'{col}_Score'] = df[col].apply(lambda x: math.log10(max(x, 1)) / math.log10(df[col].max() + 1))
    return df

ml.train_model()

if 'df_combined' not in st.session_state:
    st.session_state.df_combined = None
if 'judul_analisis' not in st.session_state:
    st.session_state.judul_analisis = ""
if 'menu_aktif' not in st.session_state:
    st.session_state.menu_aktif = "Home"

with st.sidebar:
    st.title("Menu Sistem")
    st.write("---")
    if st.button("Home", use_container_width=True):
        st.session_state.menu_aktif = "Home"
        st.rerun()
    if st.button("Upload & Proses Data", use_container_width=True):
        st.session_state.menu_aktif = "Upload & Proses Data"
        st.rerun()
    if st.button("Parameter Yang Digunakan", use_container_width=True):
        st.session_state.menu_aktif = "Parameter Yang Digunakan"
        st.rerun()

menu_pilihan = st.session_state.menu_aktif

if menu_pilihan == "Home":
    st.title("Sistem Deteksi Anomali & Risiko Pengadaan")
    st.subheader("Selamat Datang di Dashboard Analisis Fuzzy")
    st.write(
        """
        Aplikasi ini dirancang untuk mendeteksi potensi anomali atau tingkat kerawanan risiko 
        pada paket pengadaan barang dan jasa di tingkat pemerintah provinsi. 
        
        Sistem ini menggabungkan dua metode kecerdasan buatan:
        1. Machine Learning: Menganalisis tingkat urgensi berdasarkan teks nama paket.
        2. Logika Fuzzy (Sugeno & Mamdani): Mengevaluasi risiko kecurangan dengan pembobotan dinamis, 
           khususnya mendeteksi paket pengadaan bernilai fantastis namun memiliki urgensi rendah.
           
        **Sumber Dataset Resmi:** [DATA INAPROC](https://data.inaproc.id/rup)
        """
    )
    st.write("---")
    if st.session_state.df_combined is not None:
        st.success(f"Status: Data aktif termuat ({len(st.session_state.df_combined)} baris). Anda bisa langsung melihat ke halaman Hasil Analisis.")
    else:
        st.info("Status: Belum ada data aktif yang diproses. Silakan menuju ke menu Upload & Proses Data di sebelah kiri untuk memulai.")

elif menu_pilihan in ["Upload & Proses Data", "Hasil Analisis"]:
    list_menu = ["Upload & Proses Data", "Hasil Analisis"]
    tab_upload, tab_analisis = st.tabs(list_menu)
    
    with tab_upload:
        st.title("Upload File Anggaran")
        st.write("Silakan unggah satu atau beberapa berkas CSV provinsi Anda di bawah ini:")
        
        uploaded_files = st.file_uploader("Upload CSV RUP dari INAPROC", type="csv", accept_multiple_files=True)

        if uploaded_files:
            if st.button("Mulai Proses Analisis"):
                all_data = []
                provinsi = []
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                status_text.text("Langkah 1/5: Membaca dan memproses file CSV...")
                for i, uploaded_file in enumerate(uploaded_files):
                    df_processed = process_data_single_file(uploaded_file, config)
                    all_data.append(df_processed)
                    provinsi.append(df_processed['Nama Instansi'].iloc[0])
                    progress_bar.progress(((i + 1) / len(uploaded_files)) * 0.3)
                    
                df_combined = pd.concat(all_data, ignore_index=True)
            
                status_text.text("Langkah 2/5: Menjalankan deteksi urgensi dengan Machine Learning...")
                skor_ml = ml.predict_urgensi(df_combined['Nama Paket'].tolist())
                df_combined['ML_Score'] = skor_ml
                progress_bar.progress(0.5)
                
                status_text.text("Langkah 3/5: Menghitung Indeks Risiko Fuzzy Sugeno...")
                df_combined['Sugeno_Index'] = df_combined.apply(lambda r: sugeno_score(r, r['ML_Score']), axis=1)
                progress_bar.progress(0.7)

                status_text.text("Langkah 4/5: Menghitung Indeks Risiko Fuzzy Mamdani...")
                df_combined['Mamdani_Index'] = df_combined.apply(lambda r: mamdani_score(r, r['ML_Score']), axis=1)
                progress_bar.progress(0.9)

                status_text.text("Langkah 5/5: Menggabungkan hasil dan mengurutkan anomali...")
                df_combined['Combined_Index'] = (df_combined['Sugeno_Index'] + df_combined['Mamdani_Index']) / 2
                df_combined['Delta'] = abs(df_combined['Sugeno_Index'] - df_combined['Mamdani_Index'])
                df_combined['Kategori_Sugeno'] = df_combined['Sugeno_Index'].apply(klasifikasi_risiko)
                df_combined['Kategori_Mamdani'] = df_combined['Mamdani_Index'].apply(klasifikasi_risiko)
                
                df_combined = df_combined.sort_values(by='Combined_Index', ascending=False).reset_index(drop=True)
                df_combined.insert(0, 'No', range(1, len(df_combined) + 1))
                
                progress_bar.progress(1.0)
                status_text.text("Selesai! Mengarahkan ke halaman hasil analisis...")

                st.session_state.df_combined = df_combined
                
                unique_provinsi = list(set(provinsi))
                if len(unique_provinsi) == 1:
                    st.session_state.judul_analisis = f"Analisis: {unique_provinsi[0]}"
                elif len(unique_provinsi) == 2:
                    st.session_state.judul_analisis = f"Analisis: {unique_provinsi[0]} dan {unique_provinsi[1]}"
                else:
                    st.session_state.judul_analisis = f"Analisis: {unique_provinsi[0]} + {len(unique_provinsi) - 1} lainnya"
                
                st.session_state.menu_aktif = "Hasil Analisis"
                st.rerun()

    with tab_analisis:
        if st.session_state.df_combined is None:
            st.warning("Belum ada data yang diproses. Silakan masuk ke menu Upload & Proses Data terlebih dahulu.")
        else:
            st.title(st.session_state.judul_analisis)
            st.write("---")
            
            df_combined = st.session_state.df_combined
            df_display = df_combined.copy()
            
            if 'Total Nilai (Rp)' in df_display.columns:
                df_display['Total Nilai (Rp)'] = df_display['Total Nilai (Rp)'].apply(lambda x: f"{x:,.0f}")
            for col in ['Sugeno_Index', 'Mamdani_Index', 'Combined_Index', 'Delta']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")

            cols_to_show = ['No', 'Nama Instansi', 'Nama Satuan Kerja', 'Nama Paket', 'Total Nilai (Rp)', 'Sumber Dana', 'Sugeno_Index', 'Mamdani_Index', 'Delta', 'Combined_Index']
            st.write("Preview Hasil Analisis (Top 10 Anomali Tertinggi):")
            st.dataframe(df_display[cols_to_show].head(10), hide_index=True)
            avg_delta = df_combined['Delta'].mean()
            max_delta = df_combined['Delta'].max()
            min_delta = df_combined['Delta'].min()
            
            st.write("### 📊 Metrik Evaluasi Performa (Mamdani vs Sugeno)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Delta (MAD)", f"{avg_delta:.2%}")
                st.info("Konsistensi Makro Sistem")
            with col2:
                st.metric("Extreme Delta High (Max)", f"{max_delta:.2%}")
                st.error("Divergensi Kasus Ekstrem")
            with col3:
                st.metric("Extreme Delta Low (Min)", f"{min_delta:.2%}")
                st.success("Titik Konsensus Absolut")
                
            st.write("---")
            
            avg_sugeno, avg_mamdani = df_combined['Sugeno_Index'].mean(), df_combined['Mamdani_Index'].mean()
            col_s, col_m = st.columns(2)
            with col_s:
                st.metric("Sugeno Average Index", f"{avg_sugeno:.2%}")
            with col_m:
                st.metric("Mamdani Average Index", f"{avg_mamdani:.2%}")
            
            st.write("Grafik Perbandingan Tren Risiko:")
            st.line_chart(df_combined[['Sugeno_Index', 'Mamdani_Index']].sample(min(len(df_combined), 500)))
            
            st.write("Scatter Plot Deteksi Anomali (Nilai vs Risiko):")
            fig = px.scatter(
                df_combined,
                x="Total Nilai (Rp)",
                y="Combined_Index",
                color="Kategori_Mamdani",
                hover_data={
                    "Nama Instansi": True,
                    "Nama Paket": True,
                    "Total Nilai (Rp)": ":,.",
                    "Delta": ":.2%",
                    "Combined_Index": ":.2%"
                },
                labels={
                    "Total Nilai (Rp)": "Total Nilai Pengadaan", 
                    "Combined_Index": "Indeks Risiko Gabungan",
                    "Delta": "Selisih Evaluasi"
                },
                title="Peta Sebaran Paket Pengadaan"
            )
            fig.update_layout(xaxis=dict(exponentformat="none"))
            st.plotly_chart(fig, use_container_width=True)

            csv = df_combined.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil Gabungan (CSV)", csv, f"{st.session_state.judul_analisis.replace(' ', '_')}.csv", "text/csv")

elif menu_pilihan == "Parameter Yang Digunakan":
    st.subheader("6 variabel administratif yang dipetakan ke skor numerik:")
    st.markdown(
        """
        - **Metode Pengadaan**: E-Purchasing (0.3), Pengadaan Langsung (0.6), Dikecualikan (1.0)
        - **Jenis Pengadaan**: Barang (0.3), Jasa Lainnya (0.6), Jasa Konsultansi (1.0)
        - **Cara Pengadaan**: Penyedia (0.4), Swakelola (0.8)
        - **Sumber Dana**: APBD (0.3), APBDP (0.6), APBD; APBDP (1.0)
        - **Total Nilai (Rp)**: Transformasi logaritmik berbasis 10, dinormalisasi terhadap nilai maksimum dalam dataset.
        - **Skor Urgensi ML**: Skor prediksi tingkat urgensi dari model Machine Learning, diubah menjadi skor risiko dengan rumus `1 - ML_Score`.
        
        Selain pemetaan variabel di atas, sistem ini juga dilengkapi dengan **Basis Aturan Paksa (*Expert Rules*)**. 
        Komponen hibrida ini dirancang khusus untuk menangkap pola-pola anomali spesifik yang mungkin tidak sepenuhnya terdeteksi oleh model Machine Learning atau logika fuzzy standar—seperti paket pengadaan yang mengandung kata kunci sensitif atau memiliki kombinasi tidak wajar antara metode pengadaan dan sumber dana. 
        Melalui *expert rules* ini, penalti risiko tambahan akan diberikan secara selektif pada paket dengan karakteristik mencurigakan tersebut, terutama bagi proyek yang memiliki nilai nominal fantastis namun urgensi riilnya rendah. Nilai penalti ini dihitung secara dinamis, sehingga semakin besar nilai pagu proyek, semakin besar pula akumulasi potensi penalti risiko yang dijatuhkan jika paket tersebut memenuhi kriteria pelanggaran aturan.
        """
    )
    st.write("---")
    st.subheader("Visualisasi 3D dari bobot kata kunci yang dipelajari oleh model Machine Learning:")
    st.markdown(
        """
        Grafik ini menampilkan kosakata yang dipelajari oleh model Machine Learning untuk memprediksi tingkat urgensi paket pengadaan berdasarkan nama paket. 
        Setiap titik mewakili sebuah kata kunci, dengan posisi dan ukuran yang mencerminkan seberapa besar pengaruh kata tersebut terhadap prediksi model. 
        Warna titik menunjukkan apakah kata tersebut memiliki pengaruh positif (biru) atau negatif (merah) terhadap tingkat urgensi yang diprediksi.
        """
    )
    tampilkan_visualisasi_3d_ml()
