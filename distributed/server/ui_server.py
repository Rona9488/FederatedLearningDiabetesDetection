import streamlit as st
import pandas as pd
import docker
import os
import numpy as np
import torch
import json

from flower_nih.task import Net, set_weights, predict_single

# ======================
# Konfigurasi Dasar
# ======================
st.set_page_config(page_title="FL Server Dashboard", layout="centered")

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
    CONTAINERS = ["server-serverapp-1", "server-superlink-1", "server-streamlit-1"]

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

    DATA_DIR = "/app/data"
    os.makedirs(DATA_DIR, exist_ok=True)


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
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ File `{uploaded_file.name}` berhasil diunggah ke `{DATA_DIR}`")

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

    MODEL_DIR = "/app/client_checkpoints"
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".npz")])
    if not model_files:
        st.warning("Belum ada model global tersimpan.")
        st.stop()

    model_file = st.selectbox("Pilih model:", model_files)
    model_path = os.path.join(MODEL_DIR, model_file)

    st.subheader("✍️ Input Data Fitur")

    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        high_bp = int(st.checkbox("HighBP", value=False))
    with colB:
        high_chol = int(st.checkbox("HighChol", value=False))
    with colC:
        smoker = int(st.checkbox("Smoker", value=False))
    with colD:
        phys_act = int(st.checkbox("PhysActivity", value=False))
    with colE:
        diff_walk = int(st.checkbox("DiffWalk", value=False))


    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    frt = st.number_input("Fruits (porsi/hari)", min_value=0.0, value=1.0)
    veg = st.number_input("Veggies (porsi/hari)", min_value=0.0, value=1.0)
    sex = 0 if st.selectbox("Sex", ["Female", "Male"]) == "Female" else 1
    age = st.number_input("Age", min_value=1, value=30)

    input_features = [
        int(high_bp), int(high_chol), bmi,
        int(smoker), int(phys_act), frt, veg,
        int(diff_walk), 1 if sex == "Male" else 0, age
    ]

    if st.button("🔍 Jalankan Prediksi"):
        net = Net()
        w = np.load(model_path)
        set_weights(net, [w[k] for k in w.files])

        pred_class = predict_single(net, input_features)
        st.success(f"Hasil prediksi (kelas): **{pred_class}**")

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
