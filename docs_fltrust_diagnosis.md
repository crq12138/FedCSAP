# FLTrust `avg_loss` 突然变成 0 的代码级诊断

## 结论（TL;DR）
在当前实现里，`avg_loss` 变成 0 往往不是“模型真的学到完美”，而是**进入了防御逻辑/评估逻辑的边界条件**：

1. **评估样本数为 0 时，测试函数直接把 `total_l` 置 0。**
2. **FLTrust 在 `sum_trust_scores == 0` 时没有保护，权重会出现除零，导致更新失真（NaN/Inf），训练随后失效。**
3. **FLTrust 依赖最后一个客户端作为 trusted/root 更新，但主流程里 trusted 节点可能被 committee 排除，导致“参考向量”失真。**

这三个问题叠加后，在强攻击场景（尤其余弦相似度普遍 <= 0 的轮次）很容易触发“loss 突然 0 + 学不动”。

---

## 1) 为什么会看到 `avg_loss = 0`

在测试函数中，当 `dataset_size == 0` 时，代码会直接返回 0：

- `acc = ... if dataset_size!=0 else 0`
- `total_l = total_loss / dataset_size if dataset_size!=0 else 0`

也就是说，只要当前测试集/子测试集为空，就会打印 `Average loss: 0.0000`，这其实是“无样本”占位值，而不是训练正常收敛。

---

## 2) FLTrust 里真正会让训练停掉的点：`sum_trust_scores` 除零

FLTrust 主实现中：

- trust 分数来自对 trusted/root 更新的余弦相似度，并执行 `max(cos, 0)`；
- 然后直接做归一化：`... / sum_trust_scores`。

当攻击强、或者 trusted/root 参考向量不可靠时，所有客户端可能都被裁成 0 分，`sum_trust_scores == 0`，接下来会出现除零：

- 客户端权重 `wv` 归一化除零；
- `weighted_average_oracle` 里再次用 `w / tot_weights`，若 `tot_weights == 0` 会继续放大问题。

这会造成模型参数更新异常（NaN/Inf 或近似失效），随后表现为训练无法继续改善。

---

## 3) trusted/root 客户端选取逻辑存在不稳定来源

`fltrust()` 默认把**最后一个 update** 当成 clean server gradient (`clean_server_grad = grads[-1]`)。

主流程里虽然试图把 `helper.benign_namelist[-1]` 作为 trusted 节点加入训练，但如果它在 committee 中会被置空（不加入），于是 `grads[-1]` 退化为“最后一个普通客户端”，不再是稳定 root。

这会直接破坏 FLTrust 的基准方向，导致大量客户端 cosine 被压成非正值，从而触发第 2 点的 `sum_trust_scores == 0`。

---

## 建议优先修复顺序

1. **先加数值保护（必须）**
   - 在 FLTrust 中当 `sum_trust_scores <= eps` 时走 fallback（如跳过本轮、回退 FedAvg、或仅用 trusted/root 更新），禁止直接除零。
2. **显式传递 trusted/root 身份（必须）**
   - 不要依赖“最后一个客户端”隐式约定；应按 `name == trusted_id` 明确定位 root update。
3. **区分“0 loss”语义（强烈建议）**
   - `dataset_size==0` 时把日志改成 `N/A` 或单独字段，避免误判成收敛。

