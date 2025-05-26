# 🚀 Federated Learning with Flower + PyTorch

A complete Federated Learning system using [Flower](https://flower.ai/) and PyTorch, powered by Docker, Ansible, and SSH automation.

---

## 🛠️ Prerequisites

Sebelum menjalankan sistem ini, pastikan:

* Semua host (server dan client) dapat diakses via **SSH tanpa password** (menggunakan SSH key).
* Semua user host target telah:

  * Didaftarkan di `hosts.yml`
  * Memiliki akses ke `docker` **tanpa `sudo`** (misalnya dengan `sudo usermod -aG docker $USER`).
* Python 3.8+ telah terinstal.

---

## 📄 1. Konfigurasi Host: `hosts.yml`

Edit file `hosts.yml` di root proyek, dan masukkan semua informasi host yang akan digunakan sebagai server dan client:

```yaml
server:
  ip: localhost
  username: your_local_username
  ip_pub: 192.168.1.100

clients:
  - ip: 192.168.1.101
    username: clientuser1
  - ip: 192.168.1.102
    username: clientuser2
```

Pastikan setiap host:

* Dapat diakses melalui SSH tanpa password.
* Telah menginstal Docker dan memiliki izin menjalankannya tanpa sudo.

---

## 📦 2. Instalasi Dependency

Install semua dependency lokal di direktori proyek:

```bash
pip install -e .
```

---

## ⚙️ 3. Deployment Otomatis

Setelah konfigurasi `hosts.yml` dan dependensi terpasang, jalankan:

```bash
python deploy.py
```

Ini akan:

* Menghasilkan file inventori dan konfigurasi host Ansible.
* Menyalin file konfigurasi dan Docker Compose ke masing-masing host.
* Mempersiapkan dan menjalankan semua container server dan client.

---

## 🧠 4. Jalankan Federated Learning

Setelah semua container aktif, jalankan training Federated Learning dengan Deployment Engine Flower:

```bash
flwr run . --stream
```

Training akan dijalankan secara terdistribusi menggunakan klien dan server yang telah dideploy.

---

## 📌 Tips Tambahan

* Untuk regenerasi sertifikat TLS, jalankan:

  ```bash
  python deploy.py --generate-certs
  ```

* Untuk restart semua container:

  ```bash
  python deploy.py --restart
  ```

* Untuk membersihkan semua container:

  ```bash
  python deploy.py --cleanup
  ```

---

## 📚 Referensi

* 🌼 Flower website: [flower.ai](https://flower.ai/)
* 📖 Dokumentasi Flower: [flower.ai/docs](https://flower.ai/docs/)
* 🐙 GitHub Repo: [github.com/adap/flower](https://github.com/adap/flower)
* 💬 Komunitas:

  * [Slack](https://flower.ai/join-slack/)
  * [Forum Diskusi](https://discuss.flower.ai/)

---

## ⭐️ Ayo Berkontribusi!

Berikan ⭐ di repo GitHub ini jika proyek ini membantu kamu!
Pull request dan issue sangat diterima untuk pengembangan lebih lanjut.
