# Self-Play Training System for SecretMafia

这个文档介绍了一套完整的自我对练训练系统，用于优化Michael agent的动态策略池。

## 系统概述

该系统包含以下核心组件：

1. **策略池管理器** (`src/strategy_pool_manager.py`) - 智能策略选择和优化
2. **自我对练训练器** (`self_play_training.py`) - 多线程自我对练执行
3. **策略评估器** (`src/strategy_evaluator.py`) - 深度策略分析和评估
4. **训练监控面板** (`monitor_training.py`) - 实时Web监控界面

## 快速开始

### 1. 安装依赖

```bash
pip install textarena>=0.7.2
pip install streamlit plotly pandas numpy matplotlib seaborn scipy
pip install scikit-learn  # 用于相似性计算
```

### 2. 基础自我对练训练

```bash
# 运行基础训练（10个批次，每批次20场游戏）
python self_play_training.py --num-batches 10 --games-per-batch 20 --workers 4

# 高级训练配置
python self_play_training.py \
    --num-batches 20 \
    --games-per-batch 50 \
    --workers 8 \
    --strategy-pool strategy_pool.json \
    --log-dir training_logs
```

### 3. 实时监控训练

```bash
# 启动Web监控面板
streamlit run monitor_training.py --server.port 8501

# 或者直接运行（自动启动Streamlit）
python monitor_training.py --port 8501
```

### 4. 策略池评估

```bash
# 运行全面评估
python -c "
from src.strategy_evaluator import StrategyEvaluator
evaluator = StrategyEvaluator('strategy_pool.json')
results = evaluator.comprehensive_evaluation()
evaluator.create_evaluation_dashboard('strategy_dashboard.png')
evaluator.save_evaluation_report('strategy_evaluation.json')
print('评估完成！')
"
```

## 核心功能详解

### 策略池管理器 (StrategyPoolManager)

**智能策略选择：**
- 平衡探索与利用
- 基于性能的加权选择
- 多样性保证机制
- 上下文相关性匹配

**自动优化功能：**
- 淘汰表现不佳的策略
- 创建成功策略的变体
- 维护策略池多样性
- 基于角色的性能优化

### 自我对练训练器 (SelfPlayTrainer)

**并行训练执行：**
- 多线程游戏执行
- 动态策略分配
- 实时性能追踪
- 自动保存训练日志

**训练会话管理：**
- 会话状态持久化
- 训练进度监控
- 策略使用统计
- 性能改进计算

### 策略评估器 (StrategyEvaluator)

**全面性能分析：**
- 多维度性能指标
- 角色特定效果分析
- 游戏阶段效果评估
- 策略交互分析

**优化建议生成：**
- 自动识别改进机会
- 优先级排序的建议
- 预期影响评估
- 可视化报告生成

### 训练监控面板

**实时监控：**
- 策略池状态实时更新
- 训练进度追踪
- 性能指标可视化
- 自动刷新机制

**深度分析：**
- 交互式策略详情
- 训练趋势分析
- 策略对比工具
- 导出报告功能

## 训练工作流程

### 标准训练流程

1. **初始化阶段**
   ```python
   from self_play_training import SelfPlayTrainer

   trainer = SelfPlayTrainer(
       strategy_pool_path="strategy_pool.json",
       log_dir="training_logs",
       max_workers=4
   )
   ```

2. **训练执行**
   ```python
   # 运行完整训练会话
   final_report = trainer.run_training_session(
       num_batches=10,
       games_per_batch=20
   )
   ```

3. **结果分析**
   ```python
   # 生成训练报告
   report = trainer.generate_training_report(save_to_file=True)

   # 查看关键指标
   print(f"最终平均性能: {report['average_performance']:.3f}")
   print(f"总游戏数: {report['total_games_completed']}")
   ```

### 高级训练配置

**自定义训练参数：**
```python
# 创建自定义训练器
trainer = SelfPlayTrainer(
    strategy_pool_path="custom_strategy_pool.json",
    log_dir="advanced_training",
    max_workers=8
)

# 自定义训练参数
trainer.games_per_session = 100
trainer.optimization_interval = 5
trainer.min_players_per_game = 4
trainer.max_players_per_game = 7
```

**策略池调优：**
```python
from src.strategy_pool_manager import StrategyPoolManager

manager = StrategyPoolManager("strategy_pool.json")

# 调整优化参数
manager.exploration_rate = 0.3  # 增加探索
manager.performance_window = 15  # 缩短性能窗口
manager.min_usage_threshold = 3  # 降低最小使用阈值

# 运行优化
manager.optimize_strategy_pool(training_results)
```

## 性能监控和分析

### 关键指标

**策略性能指标：**
- 平均性能分数 (0-1)
- 使用次数和频率
- 成功率和胜率贡献
- 角色特定表现

**训练进度指标：**
- 累计游戏数量
- 策略池大小变化
- 平均性能趋势
- 多样性指数

**质量评估指标：**
- 整体质量评分 (A+ 到 F)
- 组件得分平衡
- 覆盖率分析
- 优化建议数量

### 可视化工具

**自动生成图表：**
- 策略性能分布直方图
- 使用频率 vs 性能散点图
- 角别性能对比图
- 训练进度时间线

**交互式仪表板：**
- 实时更新的策略排名
- 策略详情查看器
- 训练历史分析
- 导出功能

## 最佳实践

### 训练配置建议

**新手配置：**
```bash
python self_play_training.py \
    --num-batches 5 \
    --games-per-batch 10 \
    --workers 2
```

**进阶配置：**
```bash
python self_play_training.py \
    --num-batches 15 \
    --games-per-batch 30 \
    --workers 6 \
    --strategy-pool enhanced_strategy_pool.json
```

**生产级配置：**
```bash
python self_play_training.py \
    --num-batches 50 \
    --games-per-batch 100 \
    --workers 12 \
    --log-dir production_logs
```

### 监控和维护

**定期检查：**
- 每日运行策略评估
- 监控策略池质量指标
- 检查训练日志异常
- 验证策略多样性

**优化周期：**
- 短期优化：每100场游戏
- 中期评估：每500场游戏
- 长期分析：每2000场游戏

### 故障排除

**常见问题：**

1. **内存使用过高**
   ```python
   # 减少并行工作者数量
   trainer = SelfPlayTrainer(max_workers=2)

   # 增加垃圾回收频率
   import gc
   gc.collect()
   ```

2. **策略池性能下降**
   ```python
   # 重置探索率
   manager.exploration_rate = 0.4

   # 运行策略清理
   manager._prune_underperforming_strategies()
   ```

3. **训练收敛缓慢**
   ```python
   # 增加优化频率
   trainer.optimization_interval = 3

   # 调整选择阈值
   manager.similarity_threshold = 0.6
   ```

## 扩展和定制

### 添加自定义评估指标

```python
class CustomStrategyEvaluator(StrategyEvaluator):
    def _calculate_custom_metrics(self, strategies):
        # 实现自定义评估逻辑
        pass

    def comprehensive_evaluation(self):
        base_evaluation = super().comprehensive_evaluation()
        custom_metrics = self._calculate_custom_metrics(self.strategy_manager.strategies)
        base_evaluation["custom_metrics"] = custom_metrics
        return base_evaluation
```

### 自定义策略选择算法

```python
class CustomStrategyPoolManager(StrategyPoolManager):
    def select_strategies_for_training(self, num_strategies=10):
        # 实现自定义选择逻辑
        # 可以考虑游戏历史、对手策略等
        return selected_strategies
```

### 添加新的可视化图表

```python
def create_custom_visualization(data):
    # 使用Plotly或Matplotlib创建自定义图表
    fig = px.scatter_3d(
        x=data['usage'],
        y=data['performance'],
        z=data['age'],
        color=data['type']
    )
    return fig
```

## 文件结构

```
mindgames-starter-kit/
├── self_play_training.py          # 主训练脚本
├── monitor_training.py            # 监控面板
├── src/
│   ├── strategy_pool_manager.py   # 策略池管理
│   └── strategy_evaluator.py      # 策略评估
├── strategy_pool.json             # 策略池数据
├── self_play_logs/                # 训练日志目录
├── training_logs/                 # 训练会话日志
└── evaluation_results/            # 评估报告
```

## 下一步计划

1. **分布式训练支持** - 支持多机器并行训练
2. **强化学习集成** - 集成RL算法优化策略
3. **对手建模** - 建立对手策略模型
4. **自动超参数调优** - 自动优化训练参数
5. **策略导出功能** - 导出优化后的策略到其他平台

## 技术支持

如有问题或建议，请：
1. 检查训练日志文件
2. 运行策略评估诊断
3. 查看监控面板指标
4. 参考故障排除章节

通过这套完整的自我对练系统，你可以持续优化Michael agent的策略池，提升其在SecretMafia游戏中的表现。