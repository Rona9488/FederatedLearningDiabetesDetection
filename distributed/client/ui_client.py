import streamlit as st
import pandas as pd
import docker
import os
import numpy as np
import time
import json

from mini_predict import Net, set_weights, predict_single

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

DATA_ID = os.getenv("DATA_ID", "data-default")  # Ambil env variable DATA_ID
DATA_PATH = os.path.join(DATA_DIR, f"{DATA_ID}.csv")

# ======================
# Konfigurasi Dasar
# ======================
st.set_page_config(page_title="FL Client Dashboard", layout="centered")

# ======================
# Sidebar Navigasi
# ======================
with st.sidebar:
    st.markdown("### 🤖 FL Client Dashboard")
    st.markdown("Built with ❤️ using Streamlit")

    # Custom horizontal line
    st.markdown("---")

    section = st.radio(
        "📚 Navigasi",
        [
            "📦 Container Management",
            "📊 Dataset Management",
            "🧠 Prediksi Diabetes",
            "📈 Visualisasi Training"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("📍 Versi: 1.0.0")
    st.caption("© 2025 Federated Learning Project")

# ======================
# Halaman: Container Management
# ======================
if section == "📦 Container Management":
    st.title("📦 Container Management")

    client = docker.from_env()
    CONTAINERS = ["client-clientapp-1", "client-supernode-1", "client-streamlit-1"]

    for name in CONTAINERS:
        with st.container():
            st.subheader(f"🔹 `{name}`")
            try:
                container = client.containers.get(name)
                st.markdown(f"**Status:** `{container.status}`")

                col1, col2, col3, col4, col5 = st.columns([0.5, 0.1, 0.5, 0.1, 2])
                with col1:
                    if st.button("▶️ Start", key=f"start-{name}"):
                        container.start()
                        st.rerun()
                with col3:
                    if st.button("⏹️ Stop", key=f"stop-{name}"):
                        container.stop()
                        st.rerun()
                with col5:
                    if st.button("🔄 Restart", key=f"restart-{name}"):
                        container.restart()
                        st.rerun()

                with st.expander(f"🪵 Logs {name}", expanded=False):
                    logs = container.logs(tail=20).decode()
                    st.code(logs, language="bash")

            except docker.errors.NotFound:
                st.error(f"❌ Container `{name}` tidak ditemukan!")

            st.markdown("---")  # Pemisah visual antar container

# ======================
# Halaman: Dataset Management
# ======================
elif section == "📊 Dataset Management":
    st.title("📊 Dataset Management")

    # Cek apakah modal penghapusan sedang aktif
    if "show_delete_modal" not in st.session_state:
        st.session_state.show_delete_modal = False
    if "file_to_delete" not in st.session_state:
        st.session_state.file_to_delete = None

    @st.dialog("🗑️ Konfirmasi Penghapusan Dataset")
    def confirm_delete():
        selected_file = st.session_state.file_to_delete
        st.warning(f"Apakah Anda yakin ingin menghapus `{selected_file}`?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Ya, hapus"):
                os.remove(os.path.join(DATA_DIR, selected_file))
                st.success(f"✅ File `{selected_file}` berhasil dihapus.")
                st.rerun()
        with col2:
            if st.button("❌ Batal"):
                st.session_state.show_delete_modal = False
                st.session_state.file_to_delete = None
                st.rerun()


    option = st.radio("Pilih sumber data:", ("Upload Dataset Sendiri", "Gunakan Dataset yang Tersedia"))

    if option == "Upload Dataset Sendiri":
        uploaded_file = st.file_uploader("Unggah dataset Anda (CSV)", type="csv")
        if uploaded_file is not None:
            new_filename = f"{DATA_ID}.csv"
            new_file_path = os.path.join(DATA_DIR, new_filename)

            # Cek apakah file dengan nama DATA_ID.csv sudah ada
            if os.path.exists(new_file_path):
                # Rename file lama, misal tambahkan timestamp
                timestamp = int(time.time())
                old_backup_name = f"{DATA_ID}_backup_{timestamp}.csv"
                old_backup_path = os.path.join(DATA_DIR, old_backup_name)
                os.rename(new_file_path, old_backup_path)
                st.info(f"File lama `{new_filename}` diganti nama menjadi `{old_backup_name}` agar tidak ter-overwrite.")

            # Simpan file baru dengan nama DATA_ID.csv
            with open(new_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ File berhasil diunggah dan disimpan sebagai `{new_filename}` di `{DATA_DIR}`")
 
    elif option == "Gunakan Dataset yang Tersedia":
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if csv_files:
            selected_file = st.selectbox("Pilih dataset:", csv_files)
            selected_path = os.path.join(DATA_DIR, selected_file)

            try:
                df = pd.read_csv(selected_path)
                st.subheader(f"📄 Isi: `{selected_file}`")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

            # Tombol hapus dataset
            if st.button("🗑️ Hapus dataset"):
                st.session_state.file_to_delete = selected_file
                st.session_state.show_delete_modal = True
                confirm_delete()

        else:
            st.warning("⚠️ Tidak ada file dataset tersedia di folder `/app/data/`.")



# ======================
# Halaman: Prediksi Diabetes
# ======================
elif section == "🧠 Prediksi Diabetes":
    st.title("🧠 Prediksi Diabetes (Model Global)")

    BASE_MODEL_DIR = "/app/received_global_models"

    # List folder training
    training_dirs = sorted(
        [d for d in os.listdir(BASE_MODEL_DIR) if os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
    )
    if not training_dirs:
        st.info("Belum ada folder training tersimpan.")
        st.stop()

    selected_training = st.selectbox("Pilih folder training:", training_dirs)
    selected_dir_path = os.path.join(BASE_MODEL_DIR, selected_training)

    # List model di folder training yang dipilih
    model_files = sorted([f for f in os.listdir(selected_dir_path) if f.endswith(".npz")])
    if not model_files:
        st.info("Folder ini tidak memiliki model .npz.")
        st.stop()

    selected_model = st.selectbox("Pilih model:", model_files)
    model_path = os.path.join(selected_dir_path, selected_model)


    bmi_min, bmi_max = 12.0, 35.0  # default fallback
    # if os.path.exists(DATA_PATH):
    #     try:
    #         df_ref = pd.read_csv(DATA_PATH)
    #         if "BMI" in df_ref.columns:
    #             bmi_min = df_ref["BMI"].min()
    #             bmi_max = df_ref["BMI"].max()
    #     except Exception as e:
    #         st.warning(f"Gagal membaca dataset referensi untuk BMI: {e}")

    # ===== Input Fitur =====
    st.subheader("✍️ Input Data Fitur")

    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        high_bp = int(st.checkbox("HighBP"))
    with colB:
        high_chol = int(st.checkbox("HighChol"))
    with colC:
        smoker = int(st.checkbox("Smoker"))
    with colD:
        phys_act = int(st.checkbox("PhysActivity"))
    with colE:
        diff_walk = int(st.checkbox("DiffWalk"))

    colF, colG = st.columns(2)
    with colF:
        frt = int(st.checkbox("Makan Buah (rutin tiap hari)?"))
    with colG:
        veg = int(st.checkbox("Makan Sayur (rutin tiap hari)?"))

    # BMI dengan standarisasi
    raw_bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    if bmi_max != bmi_min:
        bmi_std = (raw_bmi - bmi_min) / (bmi_max - bmi_min)
    else:
        bmi_std = 0.0  # atau nilai default lain jika data tidak bervariasi
        
    # st.info(bmi_std)

    # Sex (binary)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex_val = 0 if sex == "Female" else 1

    # Usia berdasarkan rentang
    age_labels = [
        "18–24", "25–29", "30–34", "35–39",
        "40–44", "45–49", "50–54", "55–59",
        "60–64", "65–69", "70–74", "75–79",
        "80+"
    ]
    age_selection = st.selectbox("Age Range", age_labels)
    age_val = age_labels.index(age_selection) + 1  # hasil 1-13

    # Lengkapi input fitur
    input_features = [
        high_bp, phys_act, high_chol, veg, frt, bmi_std, age_val, sex_val, diff_walk, smoker  
    ]

    if st.button("🔍 Jalankan Prediksi"):
        net = Net()
        w = np.load(model_path)
        set_weights(net, [w[k] for k in w.files])

        pred_class = predict_single(net, input_features)
        if pred_class == 0:
            st.success("🎉 Aman! Kamu nggak berisiko tinggi diabetes")
            st.markdown(
                "Dari info yang kamu masukin, hasilnya sih aman-aman aja 😎. "
                "**Kamu termasuk golongan low-risk** buat diabetes. Tapi yaa, jangan jadi alasan buat santai terus!\n\n"
                "Tetap makan sehat 🥗, olahraga dikit-dikit 🏃‍♀️, dan jangan lupa tidur cukup 😴. "
                "Cek kesehatan rutin juga tetap penting yaa 🩺."
            )
        else:
            st.warning("⚠️ Waspada! Ada potensi risiko diabetes nih")
            st.markdown(
                "**Eh, hasilnya nunjukin kamu ada di kategori berisiko nih.** Tenang, ini belum tentu pasti ya, "
                "tapi udah waktunya buat lebih peduli sama gaya hidup kamu 👀.\n\n"
                "Yuk mulai pelan-pelan: kurangin makanan manis 🍰, coba lebih aktif gerak 🕺, dan kalau bisa, konsultasi ke dokter 👩‍⚕️. "
                "Langkah kecil sekarang bisa bantu banget buat nanti."
            )



# ======================
# Halaman: Visualisasi Training
# ======================
elif section == "📈 Visualisasi Training":
    st.title("📈 Visualisasi Hasil Training")

    base_path = "/app/received_global_models"
    trainings = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    if not trainings:
        st.info("Belum ada sesi training ditemukan.")
    else:
        selected_training = st.selectbox("Pilih sesi training:", trainings)
        metrics_path = os.path.join(base_path, selected_training, "metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            rounds = sorted(map(int, metrics.keys()))
            losses = [metrics[str(r)]["loss"] for r in rounds]
            accuracies = [metrics[str(r)]["accuracy"] for r in rounds]

            st.subheader("📉 Loss per Round")
            st.line_chart({"Loss": losses})

            st.subheader("📈 Accuracy per Round")
            st.line_chart({"Accuracy": accuracies})
        else:
            st.warning(f"⚠️ Tidak ditemukan `metrics.json` di sesi `{selected_training}`.")
