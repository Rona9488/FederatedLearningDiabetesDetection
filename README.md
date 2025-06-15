# 🚀 Federated Learning with Flower + PyTorch

A complete Federated Learning system using [Flower](https://flower.ai/) and PyTorch, powered by Docker, Ansible, and SSH automation.

---

## 🔒 Mengapa Federated Learning Diimplementasikan?

Data medis seperti **informasi pasien diabetes** sangat sensitif dan dilindungi oleh regulasi privasi seperti **GDPR**, sehingga **pendekatan tradisional** yang mengharuskan **pengiriman data mentah ke server pusat** berisiko tinggi terhadap kebocoran dan penyalahgunaan data. **Federated Learning (FL)** menjadi solusi dengan **melatih model langsung di tempat data berada** seperti di perangkat lokal tanpa perlu memindahkan data ke pusat. Dengan demikian, **FL membantu menjaga privasi dan mencegah kebocoran informasi**.

---

## 🖧 Bagaimana Federated Learning Diimplementasikan?

<p align="center">
  <img src="https://github.com/user-attachments/assets/8553e22e-fd79-4258-b8f1-c02e093f37e3" width="300"/>
</p>


Implementasi Federated Learning dalam proyek ini menggunakan framework Flower, yang memfasilitasi skenario federasi antara server dan klien. Arsitektur sistem terdiri dari empat komponen utama:

🖥️ **Komponen Server**
* **ServerApp**: Bertindak sebagai aggregator dan coordinator yang mengelola siklus pelatihan federatif.
* **SuperLink**: Menjembatani komunikasi antara server dan klien.

💻 **Komponen Klien**
* **ClientApp**: Melakukan pelatihan model secara lokal menggunakan data pribadi yang tersimpan di masing-masing klien.
* **SuperNode**: Mengatur komunikasi antara klien dan server.

🔁 **Proses Federated Learning**
1. **Server menginisiasi** pelatihan dengan mengirimkan model global awal ke masing-masing klien.
2. **Klien melatih model secara lokal** menggunakan data mereka sendiri tanpa mengirimkan data mentah.
3. Setelah pelatihan, klien hanya mengirimkan **parameter model (bobot)** hasil pelatihan ke server.
4. **Server mengagregasi parameter** dari semua klien untuk memperbarui model global.
5. Model global yang diperbarui dikirimkan kembali ke klien untuk iterasi berikutnya.

Dengan mekanisme ini, data tidak pernah keluar dari perangkat klien, menjadikan Federated Learning sebagai solusi nyata untuk privacy-preserving machine learning.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a65289dd-1a23-463f-94b8-385e7993db7a" width="470"/>
</p>

🌐 **Infrastruktur**
* **Server** dijalankan di virtual environment menggunakan **Proxmox**.
* **Klien** disimulasikan menggunakan layanan **Amazon EC2**.
* Semua komponen terhubung melalui jaringan privat yang aman menggunakan **VPN Tailscale**.

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
