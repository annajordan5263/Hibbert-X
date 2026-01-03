# Hibbert-X: 仿生智能化机器人算法平台

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/hibbert-x/hibbert-x)
[![Python](https://img.shields.io/badge/python-3.7+-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/cuda-11.8+-orange)](https://developer.nvidia.com/cuda-toolkit)

## 🦀 简介

Hibbert-X 是一个基于远古节肢动物 Hibbertopterus（希伯特翼鲎）的仿生智能化机器人算法平台。该平台融合了先进的 C++/MPI/CUDA 技术栈，结合 Python FastAPI API 接口，专为复杂水陆两栖环境的探索、挖掘与特殊任务而设计。

### 🎯 核心特性

- **生物启发**：基于希伯特翼鲎的演化智慧，实现高效的水陆两栖运动
- **高性能计算**：CUDA GPU 加速 + MPI 分布式计算
- **智能行为**：捕食、运动、环境适应等仿生行为算法
- **实时API**：FastAPI RESTful 接口，支持实时控制和监控
- **工业级**：支持生产环境部署，具备高可用性和扩展性

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Hibbert-X 系统架构                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Python    │  │    C++      │  │    CUDA     │         │
│  │   API       │  │   Core      │  │   Kernels   │         │
│  │   Layer     │  │   Engine    │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastAPI Web Server                   │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                               │   │
│         ▼                                               │   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Database & Storage                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 🛠️ 技术栈

- **后端**: C++17, CUDA 11.8, MPI
- **Python**: Python 3.7+, FastAPI, Pydantic
- **数据库**: PostgreSQL/MySQL, SQLAlchemy
- **容器化**: Docker, Docker Compose, Kubernetes
- **硬件**: NVIDIA GPU, 多核CPU

## 🚀 快速开始

### 环境要求

- **操作系统**: Linux Ubuntu 20.04+ (推荐)
- **GPU**: NVIDIA GPU with CUDA support
- **CPU**: 8+ cores, 16GB+ RAM
- **Python**: 3.7+

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/hibbert-x/hibbert-x.git
cd hibbert-x

# 2. 设置环境
./scripts/setup_env.sh

# 3. 构建系统
./scripts/build.sh

# 4. 启动服务
docker-compose up -d
```

### 手动安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 构建C++扩展
python setup.py build_ext --inplace

# 启动API服务
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🎮 使用示例

### API 调用示例

```bash
# 检查系统健康状态
curl http://localhost:8000/health

# 获取机器人状态
curl http://localhost:8000/api/v1/robot/state

# 控制机器人移动
curl -X POST http://localhost:8000/api/v1/robot/move \
  -H "Content-Type: application/json" \
  -d '{"target_x": 1.0, "target_y": 0.0, "target_z": 0.0}'

# 执行捕食行为
curl -X POST http://localhost:8000/api/v1/robot/hunting/execute

# 获取传感器数据
curl http://localhost:8000/api/v1/sensors/readings
```

### Python 客户端示例

```python
import requests
import json

# 连接到Hibbert-X API
base_url = "http://localhost:8000"

# 获取机器人状态
response = requests.get(f"{base_url}/api/v1/robot/state")
state = response.json()
print(f"机器人位置: {state['position']}")
print(f"能量等级: {state['energy_level']}%")

# 控制机器人移动
move_data = {
    "target_x": 2.0,
    "target_y": 1.0, 
    "target_z": 0.0
}
response = requests.post(f"{base_url}/api/v1/robot/move", json=move_data)
print(f"移动结果: {response.json()}")
```

## 🧠 仿生算法特性

### 🏊‍♂️ 运动控制
- **六足步态**: 模拟希伯特翼鲎的六足运动模式
- **水陆两栖**: 自适应水环境和陆地环境
- **扫食步态**: 特化的食物搜索和采集步态

### 🎯 捕食行为
- **多模态感知**: 视觉、声呐、化学传感器融合
- **智能追踪**: 基于预测算法的目标追踪
- **环境适应**: 根据环境条件调整捕食策略

### 🌊 物理仿真
- **流体力学**: 精确的水动力学计算
- **材料力学**: 仿生材料特性仿真
- **动力学**: 实时运动学和动力学计算

## 📊 API 接口

### 机器人控制 API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/robot/state` | GET | 获取机器人当前状态 |
| `/api/v1/robot/move` | POST | 控制机器人移动 |
| `/api/v1/robot/hunting/execute` | POST | 执行捕食行为 |
| `/api/v1/robot/environment/adapt` | POST | 环境适应 |

### 传感器 API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/sensors/readings` | GET | 获取所有传感器读数 |
| `/api/v1/sensors/status` | GET | 获取传感器状态 |
| `/api/v1/sensors/environment` | GET | 获取环境数据 |
| `/api/v1/sensors/calibrate` | POST | 校准传感器 |

### 仿真 API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/simulation/run` | POST | 运行物理仿真 |
| `/api/v1/simulation/fluid/properties` | GET | 获取流体属性 |
| `/api/v1/simulation/material/properties` | GET | 获取材料属性 |

## 🚀 部署

### Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f hibbert-x-api
```

### Kubernetes 部署

```bash
# 应用部署配置
kubectl apply -f deployment/kubernetes/

# 检查部署状态
kubectl get deployments
kubectl get services
```

### 生产环境配置

创建 `config/config.yaml` 文件：

```yaml
robot:
  name: "hibbert_x"
  max_speed: 2.0
  energy_capacity: 100.0

database:
  url: "postgresql://user:password@db:5432/hibbert_x"
  pool_size: 10

api:
  host: "0.0.0.0"
  port: 8000
  debug: false
```

## 🧪 测试

```bash
# 运行单元测试
./scripts/test.sh

# 运行Python测试
python -m pytest src/python/tests/ -v

# 运行性能测试
python -c "
import time
from hibbert_x_cpp import HibbertCore

core = HibbertCore()
start_time = time.time()
for i in range(1000):
    core.update_state(50)
end_time = time.time()

print(f'1000次状态更新耗时: {end_time - start_time:.3f}秒')
"
```

## 📚 文档

- [架构文档](docs/architecture.md)
- [用户指南](docs/user_guide.md)
- [API文档](http://localhost:8000/docs) (运行后访问)

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 GNU General Public License v3.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📞 支持

- **问题**: [GitHub Issues](https://github.com/hibbert-x/hibbert-x/issues)
- **邮件**: hibbert-x@open-source.org
- **文档**: [项目文档](docs/)

---

**Hibbert-X** - 向远古节肢动物致敬，为未来创造 🦀🔬🧠

*Powered by bio-inspired intelligence and modern computing*