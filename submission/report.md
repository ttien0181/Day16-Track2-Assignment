1. Tôi dùng AWS, us-east-1, instance t3.medium (CPU + LightGBM), source commit 55539f6.
2. Dataset creditcard.csv có 284,807 dòng, chia train/validation/test theo tỷ lệ 60/20/20, seed 42.
3. Load dữ liệu mất 2.6445 giây; training mất 1.5839 giây; best iteration là 1.
4. AUC 0.938095, Accuracy 0.99907, F1 0.764444, Precision 0.677165, Recall 0.877551 trên tập test.
5. Latency 1 dòng 1.2137 ms; throughput batch 1.000 dòng 693112.2 dòng/giấy; cách đo bằng script benchmark.py trên VM.
6. CPU/RAM/Network tôi quan sát lúc 04:23 (sau khi benchmark chạy xong) là: load average 0.00, CPU 0.0 us / 3.2 sy / 96.8 id, RAM 914Mi total / 203Mi used / 467Mi free, Network interface ens5 RX 279972472 bytes / 191279 packets; ảnh đính kèm screenshots/cpu_ram_network.png.
7. Billing tại 2026-10-04 ghi nhận $0.00 / chưa cập nhật; ước tính riêng: t3.medium trong 1 giờ ~ $0.02, chi phí thực tế phụ thuộc thời gian tồn tại và các tài nguyên đi kèm (NAT, ALB, EBS); ảnh đính kèm screenshots/billing_dashboard.png.
8. Tôi đã tải kết quả về local lúc 05:30 và chạy terraform destroy lúc 05:50; terraform state list trống, Destroy complete!.
