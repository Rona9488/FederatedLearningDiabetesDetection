#!/bin/bash

set -e

# 1. Setup virtualenv
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -q jinja2-cli pyyaml

# 2. Variabel
FLWR_VERSION="1.18.0"
PARTITION_ID="0"
NUM_PARTITIONS="1"
SUPERLINK_IP="127.0.0.1"
PROJECT_DIR=".."
CLIENT_DATA_DIR="./data"

# 3. Folder build hasil render
mkdir -p build/server
mkdir -p build/client

# 4. Render TOML
# Server
jinja2 ansible/roles/server/templates/pyproject.base.toml.j2 > build/server/pyproject.base.toml

jinja2 ansible/roles/server/templates/pyproject.flwr.toml.j2 -D superlink_ip="$SUPERLINK_IP" > build/server/pyproject.flwr.toml

# Client
jinja2 ansible/roles/client/templates/pyproject.base.toml.j2 > build/client/pyproject.base.toml

jinja2 ansible/roles/client/templates/pyproject.flwr.toml.j2 -D superlink_ip="$SUPERLINK_IP" > build/client/pyproject.flwr.toml

echo "[INFO] All TOML rendered."

# 5. Render Compose
# Client
jinja2 ansible/roles/client/templates/compose.yml.j2 \
  -D flwr_version="$FLWR_VERSION" \
  -D partition_id="$PARTITION_ID" \
  -D num_partitions="$NUM_PARTITIONS" \
  -D superlink_ip="$SUPERLINK_IP" \
  -D project_dir="$PROJECT_DIR" \
  -D client_data_dir="$CLIENT_DATA_DIR" \
  > build/client/client-compose.yml

# Server
jinja2 ansible/roles/server/templates/compose.yml.j2 \
  -D flwr_version="$FLWR_VERSION" \
  -D project_dir="$PROJECT_DIR" \
  > build/server/server-compose.yml

echo "[INFO] Compose files rendered."

# 6. Selesai
deactivate
