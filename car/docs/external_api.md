# 外部通信接口文档

小车在 `--external` 模式下通过 HTTP API 与外部系统（飞机、机械臂）协调工作。

## 启动

```bash
python3 controllers/run.py --external
```

Flask 监听 `0.0.0.0:6000`，对外暴露 3 个端点。

---

## 端点

### POST `/go` — 触发导航

发送目标点位编号，小车开始规划并执行导航。

**请求**

```json
{
  "point": 1
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `point` | int | 目标点位（1~7），7 为返回起点 |

**成功响应**（200）

```json
{ "ok": true, "point": 1 }
```

**失败响应**（400）

```json
{ "ok": false, "error": "无效点位 8" }
```

**重要**：`/go` 是非阻塞的——响应立即返回，不代表导航已完成。调用方应通过 `/status` 轮询等待完成。

---

### POST `/continue` — 放行 stay 等待

当小车执行到 `stay` 动作时，会暂停等待外部信号。调用此接口放行。

**请求**

空 body 即可，不需要 JSON。

**响应**（200）

```json
{ "ok": true }
```

**幂等**：重复调用无害；如果小车不在 stay 状态，调用会被忽略。

---

### GET `/status` — 查询状态

**响应**（200）

```json
{ "busy": true }
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `busy` | bool | `true` = 正在执行导航或等待 `/continue`；`false` = 空闲，可以发下一个 `/go` |

---

## 典型调用流程

```
飞机                     小车                       机械臂
 |                        |                           |
 |── POST /go {"point":1}→|                           |
 |←─ {"ok":true} ────────|                           |
 |                        |  [导航到点1]               |
 |                        |  [到达，精校]               |
 |                        |  [stay: 等待 /continue]    |
 |                        |                           |
 |                        |                    ←─ POST /continue
 |                        |── {"ok":true} ─────────→|
 |                        |  [rotate 90°]             |
 |                        |  [stay: 等待 /continue]    |
 |                        |                    ←─ POST /continue
 |                        |  [move_rel]               |
 |                        |  [stay: 等待 /continue]    |
 |                        |                    ←─ POST /continue
 |                        |  [rotate -90°]            |
 |                        |  [导航完成]                |
 |                        |                           |
 |── GET /status ───────→|                           |
 |←─ {"busy":false} ─────|                           |
 |                        |                           |
 |── POST /go {"point":2}→|  ...                      |
```

---

## 各点位 stay 序列

调用方需要知道每个点位有几次 `stay`（即需要发几次 `/continue`）：

| 点位 | stay 次数 | 动作序列 |
|------|----------|---------|
| 1 | 2 | correct → rotate(90°) → **stay** → move_rel(dy=-0.4) → **stay** → rotate(-90°) |
| 2 | 2 | correct → rotate(90°) → **stay** → move_rel(dy=0.4) → **stay** → rotate(-90°) |
| 3 | 2 | correct → rotate(-90°) → **stay** → move_rel(dy=0.4) → **stay** → rotate(90°) |
| 4 | 2 | correct → rotate(-90°) → **stay** → move_rel(dy=-0.4) → **stay** → rotate(90°) |
| 5 | 3 | correct → **stay** → move_rel(dy=-0.3) → **stay** → move_rel(dy=0.6) → **stay** → move_rel(dy=-0.3) |
| 6 | 2 | correct → rotate(180°) → **stay** → rotate(-180°) |
| 7 | 0 | correct（无 stay） |

**规则**：每次 `stay` 需要一次 `/continue`。机械臂在 `stay` 期间完成操作后发 `/continue`，车才会继续下一步动作。

---

## 注意事项

- **端口**：外部 API 端口固定为 **6000**，不与 Flask 底盘服务（5000）冲突。
- **串行执行**：同一时间只能执行一个点位。`/go` 在导航未完成时仍会接受并覆盖当前点位（旧任务被丢弃），调用方应通过 `/status` 确认 `busy=false` 后再发新的 `/go`。
- **超时**：导航过程中如路径规划失败或 LiDAR 不可靠，小车内部有兜底策略（紧急避险 / last_good_position），但不会无限重试。HTTP 请求本身建议设置 5 秒超时。
- **7 号点位**（返回起点）：需要在启动时成功记录了起点校正光束才有效，否则只有精校无附带动作。
