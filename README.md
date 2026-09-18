# SpectraScope · 野外光谱智能判读台

**An agentic interpreter of VNIR–SWIR (350–2500 nm) field & laboratory spectra for
geology, vegetation, and land-cover work — deterministic science first, LLM narrative second.**

> 灵感谱系：本项目的"体裁"参考了 2025 LLM Hackathon（Materials & Chemistry）获奖项目
> （CrystaLenz 之于 XRD、MixSense 之于 NMR、SKY 之于合成配方）——
> **确定性科学计算管线 + 受约束的智能体叙事**。但领域、科学内核、数据与用户体验全部
> 置于测绘遥感场景，与上述项目无任何代码或数据重叠。

---

## 它解决什么问题

野外光谱仪（ASD FieldSpec 一类）和实验室分光计每天产出大量反射率曲线，判读却依赖
专家经验：找吸收特征、比对诊断波段、排除大气水汽干扰区、给出矿物/植被/水体结论并
撰写记录。本项目把这条流程自动化为一条**可复现的物理管线**，并让 LLM 只做它擅长的事
——在测量数值之上进行**受约束的鉴别诊断叙事**：

```
上传/演示光谱 (CSV/TXT, nm 或 µm, 0-1 或 %)
   │
   ▼
① 预处理：重采样 1nm · Savitzky-Golay 平滑
   │
   ▼
② 双尺度连续统去除：全局凸包 (Clark & Roush 1984) + 局部滚动上包络
   │     （解决凸包压扁"缓坡上的宽吸收"的经典问题）
   ▼
③ 特征提取：峰值显著性算法（scipy find_peaks 同款，单调栈 O(n) 自实现）
   │     输出：中心 / 深度 / FWHM / 面积 / 不对称度 / 大气干扰区标记
   ▼
④ 端元匹配（17 类参考端元，两侧同一检测器，Tetracorder 式特征拟合）
   │     score = 0.55·诊断波段覆盖(一对一贪心) + 0.30·形状 Pearson r + 0.15·(1-未解释惩罚)
   │     大气水汽区 (1330-1480 / 1780-1980 nm) 特征双侧降权 ×0.35
   ▼
⑤ 判读报告：
   · 确定性模板报告 —— 无 LLM 也完整可用（推理链/证据/注意事项/后续建议）
   · 智能体报告 —— OpenAI 兼容端点（llama.cpp / vLLM / Zhipu / OpenAI）
                     只允许引用管线测得的波段与分数，输出结构化 JSON
   ▼
⑥ 追问对话：带本次分析上下文的多轮问答
```

## 快速开始

```bash
# Python 3.12，依赖极轻（numpy + fastapi + uvicorn + httpx + python-multipart，无 scipy/torch）
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8765
# 打开 http://127.0.0.1:8765
```

- **不配 LLM 即是完整可用的确定性判读工具**（演示样例、上传、图表、证据、报告全功能）。
- 配置智能体：右上角「智能体设置」填任意 OpenAI 兼容端点，例如
  - 本地 llama.cpp：`http://127.0.0.1:8080/v1`
  - 智谱：`https://open.bigmodel.cn/api/paas/v4` + API Key + `glm-4-flash`
- 或走环境变量：`SPECTRASCOPE_LLM_BASE_URL / SPECTRASCOPE_LLM_API_KEY / SPECTRASCOPE_LLM_MODEL`。

```bash
python -m pytest tests/ -q   # 33 项测试：解析/连续统/prominence 解析解/特征/匹配/API 全链路
```

## 界面

- **双面板光谱图（SVG 手绘，零前端依赖）**：Ⅰ 反射率 + 凸包连续统；Ⅱ 局部连续统去除
  曲线 + 检出特征标注（红点 + 波长）；大气干扰区红色斜纹。
- **候选卡**：Top-3 综合分/置信度/覆盖/形状/未解释五通道评分，证据映射逐条展示
  "观测波段 ← 库诊断波段"。
- **判读报告**：确定性模板或智能体 JSON（headline/推理链/波段点评/注意事项/后续建议）。
- **参考库浏览**：17 类端元的一键诊断波段速查。
- **影像模式（⑦）**：模拟 Sentinel-2 12 波段场景的 FCLS 全约束解混——分类图、地面真值
  对照、RMSE、NDVI/NDWI/铁染指数、7 类端元丰度图，点击任意像元查看其亚像元分解。
  多光谱像元**不做**吸收特征诊断（诚实边界：12 波段不支持，需高光谱），像元级结论为
  丰度分解；高岭石↔绿泥石在 S2 宽波段下相关 0.9996 的不可分性作为已知极限写进界面。
- **残差发现循环（本项目核心创新）**：场景中预埋了一条工作端元集里**不存在**的针铁矿化脉。
  流程：RMSE 残差热点锁定"模型解释不了的信号" → 智能体基于残差光谱形态对参考库候选做
  物理排序（无 LLM 则确定性穷举）→ 每个假设加入重解混，只有当**异常像元**的 RMSE 显著
  下降且新丰度可观时才被采纳。LLM 提出假设，物理裁决——包括赤铁矿、高岭石、明矾石在
  内的全部干扰假设都被否决，只有真异常被采纳（ΔRMSE +42%）。
- **野外采样路线规划**：丰度熵场贪心选点（混合像元信息量最大）+ 最小间隔约束 + 最近邻
  排序，输出带编号的 K 站路线与米制总长——"发现异常"到"去野外验证"的闭环。

## 参考库与科学口径

`app/data/spectral_library.json` 内置 17 类端元（高岭石、明矾石、白云母、方解石、
白云石、石膏、赤铁矿、针铁矿、绿泥石、绿帘石、绿色植被、枯草、水体、积雪、富铁土壤、
混凝土、沥青）。诊断波段位置按经典光谱遥感文献编纂（Clark et al. USGS SpecPub、
Hunt 1977、Kokaly et al. 2017 USGS DS 1035）；曲线为按诊断参数重建的理想化端元
（非实测拷贝，规避许可与体积问题），匹配基于特征而非曲线逐点比对，因此对连续统漂移
稳健。

**已知局限（诚实声明）**：单条光谱判读存在多解性；水/雪在连续统去除后本质同谱（仅剩
可见光反照率差异）；测试白名单中的光谱近孪生对为——赤铁矿/针铁矿（同为 Fe³+ 氧化物）、
绿帘石/绿泥石（同为 Fe-Mg-OH）、水/雪（同为冰水吸收）、沥青/混凝土（同为深色不透水面）。
工具以"Top-3 + 置信度 + 鉴别诊断叙事"呈现而非武断单一结论——这正是与
CrystaLenz/MixSense 一脉的"负责任判读"设计。

## 测试精度

17 端元 × 4 种子 × 3 组噪声/漂移合成盲测（共 204 条）：**top-1 ≈99%**，失误全部为上述
光谱近孪生对，正确答案 100% 在 Top-3；四个内置演示样例全部正确。独立复现（不同种子
配置）top-1 97%，结论一致。

## 结构

```
app/
  core/
    spectra.py      # Spectrum 容器、CSV/TXT 解析（nm/µm、百分数自适应）、演示样例合成
    preprocess.py   # 重采样、Savitzky-Golay（自实现免 scipy）、凸包/局部连续统
    features.py     # 峰值显著性特征提取（单调栈 O(n)）
    library.py      # 参考库、端元签名、一对一特征拟合匹配
    pipeline.py     # 编排 + 确定性模板报告
    imaging.py      # 影像模式：S2 波段重采样、场景合成、FISTA-FCLS 全约束解混
    agent.py        # LLM 判读/对话（OpenAI 兼容，失败自动回退确定性报告）
  data/spectral_library.json
  main.py           # FastAPI（/api/analyze /api/chat /api/library /api/demo-samples /api/health）
web/                # 无构建、零依赖前端（SVG 图表手绘）
tests/              # 33 项 pytest
```

## Roadmap（黑客松现场可扩展方向）

- ~~Sentinel-2 场景像元级解混与丰度制图~~ ✅ 已实现（影像模式：FCLS + 指数图 + 像元点击解译）
- ~~残差驱动的异常端元发现 + 采样路线规划~~ ✅ 已实现（LLM 提出假设、物理裁决的发现循环）
- 真实 USGS Speclib / ECOSTRESS 子集接入（曲线级端元 + 元数据过滤）；真实 S2 GeoTIFF 接入
- 判读报告一键导出野外记录簿格式（GeoJSON/SHP 采样点回填）
- 多次测量不确定度传递与迁移学习微调（LoRA 领域自适应）
