# KuaiRand-Pure 数据目录说明

## 1. 官方下载与解压（推荐在开发机上执行）

```bash
cd /home/yujiawei/pre_learning/data  # 或本机 pre_learning/data
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
# 可选校验 MD5：0820331067a3784d9691136f772b35a7
tar -xzvf KuaiRand-Pure.tar.gz
```

解压后目录结构通常为：
```text
data/KuaiRand-Pure/data/
├── log_standard_4_08_to_4_21_pure.csv
├── log_standard_4_22_to_5_08_pure.csv
├── log_random_4_22_to_5_08_pure.csv
├── user_features_pure.csv
├── video_features_basic_pure.csv
└── video_features_statistic_pure.csv
```

> 说明：本机网络访问 Zenodo 官方链接返回了 `403 Forbidden (unusual traffic)` 限流，因此 `scripts/kuairand_loader.py` 内置了**自动切换机制**：
> - 如果检测到上述真实 CSV 文件，自动加载真实 `KuaiRand-Pure`；
> - 如果尚未下载，自动生成严格对齐 KuaiRand-Pure Schema 与因果曝光机制的仿真数据，让所有实验脚本开箱即跑；你在开发机拉取真实数据后，直接把 `scripts/` 拷过去即可零改动在真实数据上复现！
