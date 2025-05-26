#!/bin/sh

# Tunggu 5 detik atau tunggu supernode port terbuka
echo "[INFO] Waiting for supernode to be ready..."
sleep 5

# Jalankan flwr-clientapp dengan argumen yang diteruskan dari `command:` di compose
exec flwr-clientapp "$@"
