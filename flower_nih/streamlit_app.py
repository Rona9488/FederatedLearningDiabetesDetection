import streamlit as st
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from task import Net, train, test, get_weights, set_weights

# === PROMETHEUS METRICS SETUP ===
import threading
from prometheus_client import start_http_server, Gauge, REGISTRY

# Fungsi bantu untuk hindari duplikat metric
def get_or_create_gauge(name, description):
    try:
        return REGISTRY._names_to_collectors[name]
    except KeyError:
        return Gauge(name, description)

TRAIN_DURATION = get_or_create_gauge("train_duration_seconds", "Training duration in seconds")
VAL_LOSS = get_or_create_gauge("val_loss", "Validation loss after training")
VAL_ACCURACY = get_or_create_gauge("val_accuracy", "Validation accuracy after training")
TEST_LOSS = get_or_create_gauge("test_loss", "Loss on test data")
TEST_ACCURACY = get_or_create_gauge("test_accuracy", "Accuracy on test data")

# Jalankan server Prometheus hanya sekali saat Streamlit reload
if 'prometheus_server_started' not in st.session_state:
    threading.Thread(target=lambda: start_http_server(8000), daemon=True).start()
    st.session_state.prometheus_server_started = True
# ================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Federated Learning Client UI")
st.write("Upload your dataset (CSV), then train or test a model locally.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully!")
    st.write("Preview:", df.head())

    # Input nama kolom target
    target_column = st.text_input("Masukkan nama kolom target (label)", value="Diabetes_012")

    if target_column not in df.columns:
        st.error(f"Kolom '{target_column}' tidak ditemukan dalam dataset!")
    else:
        # Preprocessing
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=16)

        # Initialize model
        model = Net()
        model.to(DEVICE)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Train Locally"):
                st.write("Training...")
                import time
                start_time = time.time()

                result = train(model, trainloader, testloader, epochs=5, learning_rate=0.001, device=DEVICE)

                duration = time.time() - start_time
                TRAIN_DURATION.set(duration)
                VAL_LOSS.set(result['val_loss'])
                VAL_ACCURACY.set(result['val_accuracy'])

                st.success("Training completed.")
                st.write(f"Val Loss: {result['val_loss']:.4f}")
                st.write(f"Val Accuracy: {result['val_accuracy']:.4f}")

        with col2:
            if st.button("Test Model"):
                loss, acc = test(model, testloader, DEVICE)

                TEST_LOSS.set(loss)
                TEST_ACCURACY.set(acc)

                st.success("Testing completed.")
                st.write(f"Test Loss: {loss:.4f}")
                st.write(f"Test Accuracy: {acc:.4f}")

