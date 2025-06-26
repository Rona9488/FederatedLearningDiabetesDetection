import os
import subprocess
import sys
import yaml
import shutil
import argparse

def check_or_install_ansible():
    try:
        subprocess.run(["ansible", "--version"], check=True, stdout=subprocess.DEVNULL)
        print("✅ Ansible already installed.")
    except subprocess.CalledProcessError:
        print("❌ Ansible not found. Installing via pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ansible"], check=True)
        print("✅ Ansible installed.")

def read_input_from_file(filepath="hosts.yml"):
    if not os.path.exists(filepath):
        print(f"❌ File {filepath} tidak ditemukan.")
        sys.exit(1)

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    server = data["server"]
    server_ip = server.get("ip")
    server_ip_pub = server.get("ip_pub")  # bisa None
    server_user = server.get("username")

    clients = []
    for client in data.get("clients", []):
        ip = client.get("ip")
        ip_pub = client.get("ip_pub")
        username = client.get("username")
        clients.append({"ip": ip, "ip_pub": ip_pub, "username": username})

    local = data.get("local")
    if local:
        local_ip = local.get("ip")
        local_user = local.get("username")
    else:
        local_ip = None
        local_user = None

    return server_ip, server_ip_pub, server_user, clients, local_ip, local_user

def generate_inventory(server_ip, server_user, clients, local_ip=None, local_user=None):
    inventory = ""

    inventory += "[local]\n"
    inventory += f"local1 ansible_connection=local\n\n"

    inventory += "[server]\n"
    if server_ip in ["localhost", "127.0.0.1"]:
        inventory += f"server1 ansible_connection=local ansible_user={server_user}\n"
    else:
        inventory += f"server1 ansible_host={server_ip} ansible_user={server_user}\n"

    inventory += "\n[clients]\n"
    for i, client in enumerate(clients):
        hostname = f"client_{i+1}"
        ip = client["ip"]
        user = client["username"]
        if ip in ["localhost", "127.0.0.1"]:
            inventory += f"{hostname} ansible_connection=local ansible_user={user}\n"
        else:
            inventory += f"{hostname} ansible_host={ip} ansible_user={user}\n"

    os.makedirs("ansible", exist_ok=True)
    with open("ansible/inventory.ini", "w") as f:
        f.write(inventory)
    print("✅ File inventory berhasil dibuat.")

def generate_host_vars(server_ip, server_ip_pub, server_user, clients):
    vars_dir = os.path.join("ansible", "host_vars")
    os.makedirs(vars_dir, exist_ok=True)

    ui_base = 8501
    prom_base = 9200

    is_server_local = server_ip in ("localhost", "127.0.0.1")
    ip_for_vars = server_ip_pub if is_server_local and server_ip_pub else server_ip

    # Deteksi client lokal
    local_client_indices = [i for i, c in enumerate(clients) if c["ip"] in ("localhost", "127.0.0.1")]
    num_local_clients = len(local_client_indices)

    # === Tentukan mode perubahan port ===
    prom_port_should_change = (
        (is_server_local and num_local_clients >= 1) or
        (not is_server_local and num_local_clients >= 2)
    )
    ui_port_should_change = (
        (not is_server_local and num_local_clients >= 2) or
        (is_server_local and num_local_clients >= 2)
    )

    # Tulis untuk local dan server
    for target in ("local1", "server1"):
        with open(f"{vars_dir}/{target}.yml", "w") as f:
            yaml.safe_dump({"superlink_ip": ip_for_vars}, f)

    for i, client in enumerate(clients):
        client_name = f"client_{i+1}"
        vars_path = os.path.join(vars_dir, f"{client_name}.yml")

        ip_client = client["ip"]
        ip_client_pub = client.get("ip_pub")
        ip_for_client = ip_client_pub if ip_client in ("localhost", "127.0.0.1") and ip_client_pub else ip_client
        is_client_local = ip_client in ("localhost", "127.0.0.1")

        if is_client_local:
            local_id = local_client_indices.index(i)
            prom_port = prom_base + local_id + 1 if prom_port_should_change else prom_base
            ui_port = ui_base + local_id if ui_port_should_change else ui_base
        else:
            prom_port = prom_base
            ui_port = ui_base

        vars_content = {
            "partition_id": i,
            "num_partitions": len(clients),
            "superlink_ip": ip_for_vars,
            "client_ip": ip_for_client,
            "client_user": client["username"],
            "PROM_PORT": prom_port,
            "UI_PORT": ui_port
        }

        with open(vars_path, "w") as f:
            yaml.safe_dump(vars_content, f)

    print("✅ Host vars berhasil dibuat (sesuai semua aturan port lokal/remote).")



def generate_certificates(server_ip):
    try:
        certs_path = "distributed/superlink-certificates"
        if os.path.exists(certs_path):
            shutil.rmtree(certs_path)
        os.makedirs(certs_path, exist_ok=True)

        env = os.environ.copy()
        env["SUPERLINK_IP"] = server_ip

        subprocess.run(
            [
                "sudo", "docker", "compose", "-f", "distributed/certs.yml", "run", "--rm",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "gen-certs"
            ],
            check=True,
            env=env
        )
        print("✅ Sertifikat berhasil dibuat.")
    except subprocess.CalledProcessError:
        print("❌ Gagal membuat sertifikat. Pastikan docker compose berjalan dengan benar.")
        sys.exit(1)

def run_playbook(playbook):
    print(f"🚀 Menjalankan playbook: {playbook}")
    subprocess.run(["ansible-playbook", "-i", "ansible/inventory.ini", f"ansible/{playbook}"])

def test_connectivity():
    print("🧪 Testing SSH/Ansible connectivity to all hosts...")
    # contoh test: ping semua host
    subprocess.run(["ansible", "-i", "ansible/inventory.ini", "all", "-m", "ping"])

def main():
    parser = argparse.ArgumentParser(description="Deploy automation with Ansible")
    parser.add_argument("--file", type=str, default="hosts.yml", help="Path ke file hosts.yml")
    parser.add_argument("--generate-inventory", action="store_true", help="Generate Ansible inventory from hosts.yml")
    parser.add_argument("--generate-host-vars", action="store_true", help="Generate host_vars for Ansible")
    parser.add_argument("--generate-certs", action="store_true", help="Generate SSL certificates")
    parser.add_argument("--deploy", action="store_true", help="Run Ansible playbooks for server and clients")
    parser.add_argument("--test", action="store_true", help="Test SSH/Ansible connectivity to all targets")
    parser.add_argument("--build", action="store_true", help="Check target's container and build it") 
    parser.add_argument("--cleanup", action="store_true", help="Down containers and remove it") 
    parser.add_argument("--local", action="store_true", help="Run only playbook-local.yml")  # <-- Tambahkan ini
    args = parser.parse_args()

    server_ip, server_ip_pub, server_user, clients, local_ip, local_user = read_input_from_file(args.file)

    # Jika tidak ada argumen, jalankan semua
    if not any([args.generate_inventory, args.generate_host_vars, args.generate_certs, args.deploy, args.test, args.build, args.cleanup, args.local]):
        args.generate_inventory = True
        args.generate_host_vars = True
        args.local = True
        args.generate_certs = True
        args.deploy = True
        args.build = True
        # args.cleanup = True

    if args.generate_inventory:
        generate_inventory(server_ip, server_user, clients, local_ip, local_user)

    if args.generate_host_vars:
        generate_host_vars(server_ip, server_ip_pub, server_user, clients)
        
    if args.local:
        run_playbook("playbook-local.yml")
        
    if args.generate_certs:
        ip_for_cert = server_ip_pub if (server_ip in ["localhost", "127.0.0.1"] and server_ip_pub) else server_ip
        generate_certificates(ip_for_cert)

    if args.deploy:
        run_playbook("playbook-local.yml")
        run_playbook("playbook-server.yml")
        run_playbook("playbook-client.yml")

    if args.test:
        test_connectivity()

    if args.build:
        run_playbook("playbook-build.yml")
    
    if args.cleanup:
        run_playbook("playbook-cleanup.yml")
    
if __name__ == "__main__":
    main()
