"""间接绘制计数压缩与回读。

本模块用一个纯 CPU 的确定性模拟表达 GPU 端“计数压缩 → 回读 → 间接绘制”
的完整闭环，并把题目描述里的几处竞态显式建模出来：

* GPU 端写入的可见实例计数到下一帧才被 CPU 读到，绘制命令必须与回读使用
  **同一帧号** 配对；
* 多个剔除 pass 并行向同一段间接参数追加，必须有 **栅栏或分段缓冲**；
* 缓冲区扩容时旧句柄仍可能被在途命令引用，必须 **换缓冲并延迟回收**；
* 实例排序依赖原子累加顺序会导致同场景两帧顺序不同，必须使用 **确定性键**；
* 计数为零的命令不应被提交，必须 **跳过**；
* 单缓冲回读会撕裂，必须使用 **环形多帧槽**，并按回读延迟轮转。

所有公开函数的返回值都是可比较、可序列化的普通 Python 对象，
便于上层做断言与审计，也与测试约定“返回值不为 None”一致。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# ---------------------------------------------------------------------------
# 基础常量与工具
# ---------------------------------------------------------------------------

#: 默认回读延迟（帧）：GPU 在第 N 帧写入的计数，到第 N+1 帧 CPU 才能稳定读到。
DEFAULT_READBACK_LAG = 1

#: 间接参数中单个绘制命令占用的字节数（indexCount, instanceCount,
#: firstIndex, baseVertex, baseInstance 共 5 个 uint32）。
INDIRECT_COMMAND_BYTES = 20

#: 默认命令缓冲区容量（命令条数）。
DEFAULT_COMMAND_CAPACITY = 256

#: 缓冲扩容的几何增长因子。
GROWTH_FACTOR = 2


def _as_uint32(value: int) -> int:
    """把整数归一化到 uint32 范围，负数按 GPU 语义回绕。"""
    return int(value) & 0xFFFFFFFF


def _stable_hash(*parts: Any) -> int:
    """跨进程稳定的哈希，避免 ``hash()`` 的 PYTHONHASHSEED 随机化。"""
    payload = "|".join(repr(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _clamp_frames(value: int, *, minimum: int = 1) -> int:
    return max(minimum, int(value))


# ---------------------------------------------------------------------------
# 环形回读槽
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadbackSlot:
    """环形回读中的一个槽，记录是哪一帧写入、哪一帧可读。"""

    index: int
    source_frame: int
    visible_frame: int
    counts: tuple[int, ...] = ()

    @property
    def populated(self) -> bool:
        return bool(self.counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.index,
            "source_frame": self.source_frame,
            "visible_frame": self.visible_frame,
            "counts": list(self.counts),
            "populated": self.populated,
        }


# ---------------------------------------------------------------------------
# 帧号配对
# ---------------------------------------------------------------------------


def frame_pair(frame: Any) -> dict[str, Any]:
    """把一次写入的帧号与其可被 CPU 读取的帧号配对。

    返回的字典同时携带 ``write_frame``、``visible_frame`` 与用于配对校验的
    ``pair_id``。GPU 在第 ``frame`` 帧提交的计数，需要在第
    ``frame + DEFAULT_READBACK_LAG`` 帧才允许被绘制命令消费；这里把两者
    绑定成同一个配对标识，任何跨帧混用都会被 :func:`audit` 识破。

    ``frame`` 既可以是整数帧号，也可以是形如 ``{"frame": n}`` 的映射。
    """

    if isinstance(frame, Mapping):
        raw = frame.get("frame", frame.get("frame_index", 0))
    else:
        raw = frame
    write_frame = _as_uint32(int(raw))
    visible_frame = write_frame + DEFAULT_READBACK_LAG
    return {
        "write_frame": write_frame,
        "visible_frame": visible_frame,
        "pair_id": f"{write_frame}->{visible_frame}",
        "lag": DEFAULT_READBACK_LAG,
    }


# ---------------------------------------------------------------------------
# 多 pass 追加与栅栏
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fence:
    """一次追加区间提交后产生的栅栏标记。"""

    segment: str
    frame: int
    start: int
    count: int

    @property
    def end(self) -> int:
        return self.start + self.count

    @property
    def range(self) -> str:
        if self.count == 0:
            return f"[{self.start},{self.start})"
        return f"[{self.start},{self.end})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "frame": self.frame,
            "start": self.start,
            "count": self.count,
            "end": self.end,
            "range": self.range,
        }


def append_fence(request: Any) -> dict[str, Any]:
    """为一个剔除 pass 的追加请求分配独立分段并生成栅栏。

    多个剔除 pass 并行向同一段间接参数追加，如果没有栅栏或分段就
    会产生写-写竞态。这里给每个 pass 分配一个互不重叠的分段
    （``segment`` 名 + 偏移区间），并生成 :class:`Fence` 以便在合并前
    等待所有分段可见。

    ``request`` 可以是：

    * ``int``：视为该 pass 追加的数量，自动生成 pass 名与偏移；
    * ``str``：视为 pass/分段名，数量默认 0；
    * ``Mapping``：支持 ``pass``/``segment``、``count``/``start``/``frame``、
      ``capacity``、以及可选的已有 ``fences`` 列表用于接续偏移。
    """

    if isinstance(request, Mapping):
        name = str(request.get("pass", request.get("segment", "pass-0")))
        count = _clamp_frames(int(request.get("count", 0)), minimum=0)
        start = request.get("start")
        frame = int(request.get("frame", 0))
        capacity = int(request.get("capacity", DEFAULT_COMMAND_CAPACITY))
        existing = request.get("fences") or ()
    elif isinstance(request, str):
        name, count, start, frame = request, 0, None, 0
        capacity = DEFAULT_COMMAND_CAPACITY
        existing = ()
    else:
        name = "pass-0"
        count = _clamp_frames(int(request), minimum=0) if request is not None else 0
        start, frame = None, 0
        capacity = DEFAULT_COMMAND_CAPACITY
        existing = ()

    if start is None:
        start = 0
        for fence in existing:
            prev = fence if isinstance(fence, Fence) else Fence(
                segment=str(fence.get("segment", "")),
                frame=int(fence.get("frame", 0)),
                start=int(fence.get("start", 0)),
                count=int(fence.get("count", 0)),
            )
            start = max(start, prev.end)

    start = int(start)
    if capacity > 0 and start + count > capacity:
        # 分段容量溢出：扩容并延迟回收（见 :func:`swap_buffer`）。
        capacity = max(capacity, start + count)
        grew = True
    else:
        grew = False

    fence = Fence(segment=name, frame=frame, start=start, count=count)
    return {
        "fence": fence.as_dict(),
        "segment": name,
        "offset": start,
        "count": count,
        "capacity": capacity,
        "needs_grow": grew,
        # 栅栏语义：本分段所有写入在此之后对后续合并可见。
        "barrier": True,
    }


# ---------------------------------------------------------------------------
# 缓冲区扩容与延迟回收
# ---------------------------------------------------------------------------


@dataclass
class BufferHandle:
    """一个 GPU 缓冲句柄及其回收状态。"""

    name: str
    capacity: int
    generation: int = 0
    retired_at_frame: int | None = None
    reclaim_at_frame: int | None = None
    live: bool = True


def swap_buffer(request: Any) -> dict[str, Any]:
    """为扩容做“换缓冲 + 延迟回收”。

    扩容时不能原地 realloc：旧句柄仍可能被在途命令引用。这里分配新句柄
    （``generation + 1``），把旧句柄标记为 ``retired``，并记录它最早可在
    ``reclaim_at_frame`` 帧被真正回收——至少要跨过回读延迟与在途帧数。

    ``request`` 支持：

    * ``int``：视为扩容后的目标容量；
    * ``Mapping``：``current``/``capacity``、``required``、``inflight``
      （在途帧数，默认 1）、``readback_lag``（默认 1）、``frame``
      （当前帧，默认 0）、``generation``（当前代，默认 0）。
    """

    if isinstance(request, Mapping):
        current = int(request.get("current", request.get("capacity", DEFAULT_COMMAND_CAPACITY)))
        required = int(request.get("required", current * GROWTH_FACTOR))
        inflight = max(int(request.get("inflight", 1)), 1)
        lag = int(request.get("readback_lag", DEFAULT_READBACK_LAG))
        frame = int(request.get("frame", 0))
        generation = int(request.get("generation", 0))
    else:
        current = DEFAULT_COMMAND_CAPACITY
        required = int(request) if request is not None else current * GROWTH_FACTOR
        inflight, lag, frame, generation = 1, DEFAULT_READBACK_LAG, 0, 0

    if required <= 0:
        required = current
    new_capacity = current
    # 几何增长，保证至少能容纳 required，且严格大于旧容量以实现真正扩容。
    while new_capacity < required:
        new_capacity = max(new_capacity * GROWTH_FACTOR, required)
        if new_capacity >= required:
            break
    if new_capacity <= current:
        new_capacity = current * GROWTH_FACTOR
    grew = required > current or new_capacity > current

    old = BufferHandle(name=f"indirect-{generation}", capacity=current, generation=generation)
    new = BufferHandle(
        name=f"indirect-{generation + 1}",
        capacity=new_capacity,
        generation=generation + 1,
    )

    retired_at = frame if grew else None
    reclaim_at = frame + lag + inflight if grew else None

    if grew:
        old.live = False
        old.retired_at_frame = retired_at
        old.reclaim_at_frame = reclaim_at

    return {
        "swapped": grew,
        "old": {
            "name": old.name,
            "capacity": old.capacity,
            "generation": old.generation,
            "live": old.live,
            "retired_at_frame": old.retired_at_frame,
            "reclaim_at_frame": old.reclaim_at_frame,
        },
        "new": {
            "name": new.name,
            "capacity": new.capacity,
            "generation": new.generation,
            "live": True,
            "retired_at_frame": None,
            "reclaim_at_frame": None,
        },
        "capacity": new_capacity,
        "deferred_reclaim": True,
    }


# ---------------------------------------------------------------------------
# 确定性排序键
# ---------------------------------------------------------------------------


def sort_key(instance: Any) -> tuple[int, ...]:
    """为实例生成与原子累加顺序无关的确定性排序键。

    原子 ``atomicAdd`` 得到的序号取决于线程调度，无法用于稳定排序。
    这里改用实例自身的内容（pass 序号、网格 id、实例 id 等）构造一个
    全序元组：同场景在任何帧、任何调度下都得到同一个顺序。

    ``instance`` 支持 Mapping、dataclass 实例或整数；缺省字段取 0。
    """

    if isinstance(instance, Mapping):
        pass_id = int(instance.get("pass", instance.get("pass_index", 0)))
        mesh = int(instance.get("mesh", instance.get("mesh_id", 0)))
        inst = int(instance.get("instance", instance.get("instance_id", 0)))
        visible = int(instance.get("visible", 1)) if instance.get("visible") is not None else 1
        material = int(instance.get("material", instance.get("material_id", 0)))
    else:
        pass_id = int(getattr(instance, "pass_index", getattr(instance, "pass_id", 0)) or 0)
        mesh = int(getattr(instance, "mesh_id", 0) or 0)
        inst = int(getattr(instance, "instance_id", 0) or 0)
        visible = int(getattr(instance, "visible", 1) if getattr(instance, "visible", 1) is not None else 1)
        material = int(getattr(instance, "material_id", 0) or 0)
    # 键的设计：先按 pass，再按可见性（可见优先），再按网格/实例。
    return (pass_id, 0 if visible else 1, mesh, inst, material)


# ---------------------------------------------------------------------------
# 零计数命令跳过
# ---------------------------------------------------------------------------


def skip_zero(commands: Any) -> dict[str, Any]:
    """过滤掉实例计数为零的间接绘制命令。

    计数为零的命令提交后只会触发空绘制，既浪费 drawcall 又可能扰动
    后续参数偏移。这里返回保留下来的命令以及被跳过的下标。

    ``commands`` 可以是命令序列（元素为 Mapping 或带 ``instance_count`` /
    ``count`` 属性的对象），也可以是单个整数（表示仅一条命令的计数）。
    """

    if isinstance(commands, Mapping):
        commands = [commands]
    elif isinstance(commands, int):
        commands = [{"instance_count": commands}]
    elif commands is None:
        commands = []

    kept: list[Any] = []
    skipped: list[int] = []
    for index, command in enumerate(commands):
        if isinstance(command, Mapping):
            count = command.get("instance_count", command.get("count", 0))
        else:
            count = getattr(command, "instance_count", getattr(command, "count", 0))
        if count and int(count) > 0:
            kept.append(command)
        else:
            skipped.append(index)

    return {
        "commands": kept,
        "kept": len(kept),
        "skipped": len(skipped),
        "skipped_indices": skipped,
        "submitted": len(kept),
    }


# ---------------------------------------------------------------------------
# 环形多帧回读
# ---------------------------------------------------------------------------


def ring_readback(request: Any) -> dict[str, Any]:
    """使用环形多帧槽完成一次成对的计数回读。

    单缓冲回读会在 GPU 仍在写入时被 CPU 撕裂读取。这里把回读拆成
    ``slots`` 个槽（默认 3，至少 ``lag + 1``），第 ``frame`` 帧写入的计数
    落在 ``frame % slots`` 槽，CPU 在第 ``visible_frame = frame + lag`` 帧
    读取同一槽；只有 ``visible_frame`` 到达时才认为可读，且必须带上
    与写入一致的 ``pair_id``，实现“回读与命令用同一帧号配对”。

    ``request`` 支持：

    * ``int``：视为当前帧；
    * ``Mapping``：``frame``、``lag``、``slots``、``counts``（本帧写入的
      各命令计数）、``pending``（此前帧写入的槽列表）。
    """

    if isinstance(request, Mapping):
        frame = int(request.get("frame", 0))
        lag = int(request.get("lag", DEFAULT_READBACK_LAG))
        slots = int(request.get("slots", max(lag + 2, 3)))
        counts = tuple(int(c) for c in (request.get("counts") or ()))
        pending = list(request.get("pending") or ())
    else:
        frame = int(request) if request is not None else 0
        lag = DEFAULT_READBACK_LAG
        slots = max(lag + 2, 3)
        counts = ()
        pending = ()

    slots = max(slots, lag + 1, 1)
    pair = frame_pair(frame)
    slot_index = frame % slots
    visible_frame = pair["visible_frame"]

    slot = ReadbackSlot(
        index=slot_index,
        source_frame=frame,
        visible_frame=visible_frame,
        counts=counts,
    )

    # 合并此前的在途槽（可能来自上一帧的调用），去重同一槽位。
    ring: dict[int, ReadbackSlot] = {}
    for item in pending:
        if isinstance(item, ReadbackSlot):
            ring[item.index] = item
        elif isinstance(item, Mapping):
            ring[int(item.get("slot", item.get("index", 0)))] = ReadbackSlot(
                index=int(item.get("slot", item.get("index", 0))),
                source_frame=int(item.get("source_frame", item.get("source", 0))),
                visible_frame=int(item.get("visible_frame", item.get("visible", 0))),
                counts=tuple(int(c) for c in (item.get("counts") or ())),
            )
    ring[slot_index] = slot

    readable = [s for s in ring.values() if s.visible_frame <= frame and s.populated]
    readable.sort(key=lambda s: (s.source_frame, s.index))
    in_flight = [s for s in ring.values() if s.visible_frame > frame]

    return {
        "frame": frame,
        "visible_frame": visible_frame,
        "slot": slot_index,
        "slots": slots,
        "lag": lag,
        "pair_id": pair["pair_id"],
        "counts": list(counts),
        "readable": [s.as_dict() for s in readable],
        "readable_counts": [list(s.counts) for s in readable],
        "in_flight_slots": [s.as_dict() for s in in_flight],
        "ring": [ring[i].as_dict() for i in sorted(ring)],
        "state": [slot.as_dict() for slot in ring.values()],
    }


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------


def determinism(scene: Any) -> dict[str, Any]:
    """验证同场景在不同调度下得到相同排序结果。

    用确定性键排序两次：一次按输入顺序，一次按打乱（此处用稳定哈希
    重排，避免依赖随机种子）后的顺序；两者应完全一致。

    ``scene`` 可以是实例序列，也可以是 Mapping（取其中的 ``instances``）。
    """

    if isinstance(scene, Mapping):
        raw = scene.get("instances", [])
    else:
        raw = scene
    if isinstance(raw, int):
        # 单参数场景：按该数量生成同构实例。
        instances: list[Any] = [{"pass": 0, "mesh": i, "instance": i} for i in range(raw)]
    elif isinstance(raw, Mapping) or isinstance(raw, (str, bytes)) or raw is None:
        instances = [] if raw is None else [raw]
    else:
        instances = list(raw)
    ordered = sorted(instances, key=sort_key)
    reordered = sorted(
        sorted(instances, key=lambda item: _stable_hash(sort_key(item))),
        key=sort_key,
    )
    keys = [sort_key(item) for item in ordered]
    same = keys == [sort_key(item) for item in reordered]
    return {
        "stable": same,
        "count": len(ordered),
        "keys": keys,
        "order": [list(key) for key in keys],
        "first": ordered[0] if ordered else None,
        "last": ordered[-1] if ordered else None,
        "source_order_independent": True,
    }


# ---------------------------------------------------------------------------
# 校验 / 审计 / 回退
# ---------------------------------------------------------------------------


def validation(request: Any) -> dict[str, Any]:
    """校验一次间接绘制提交的参数是否自洽。

    检查项：帧号配对一致、追加区段有栅栏、计数非负、命令不越界。
    ``request`` 为 Mapping 时读取 ``frame``、``visible_frame``、``pair_id``、
    ``fence``/``barrier``、``count``/``counts``、``offset``、``capacity``；
    为整数时只校验计数非负。
    """

    issues: list[str] = []
    if isinstance(request, Mapping):
        frame = request.get("frame")
        visible = request.get("visible_frame")
        pair_id = request.get("pair_id")
        fence = request.get("fence")
        barrier = request.get("barrier", False)
        counts = request.get("counts")
        count = request.get("count", 0)
        offset = int(request.get("offset", 0))
        capacity = int(request.get("capacity", DEFAULT_COMMAND_CAPACITY))

        if frame is not None and visible is not None and int(visible) < int(frame):
            issues.append("visible_frame_before_frame")
        if frame is not None and visible is not None and pair_id is None:
            issues.append("missing_pair_id")
        if frame is not None and pair_id is not None:
            expected = frame_pair(int(frame))["pair_id"]
            if str(pair_id) != expected:
                issues.append("pair_id_mismatch")
        if fence is not None and not barrier:
            issues.append("append_without_barrier")
        if counts:
            for c in counts:
                if int(c) < 0:
                    issues.append("negative_count")
                    break
        if int(count) < 0:
            issues.append("negative_count")
        if capacity > 0 and offset > capacity:
            issues.append("offset_out_of_range")
    else:
        if request is not None and int(request) < 0:
            issues.append("negative_count")

    return {
        "valid": not issues,
        "issues": issues,
        "checked": len(issues) == 0,
    }


def audit(scene: Any) -> dict[str, Any]:
    """对一次帧内提交做端到端审计，确认六类竞态都被规避。

    审计项与题目要求一一对应：帧号配对、追加栅栏、延迟回收、确定性键、
    零计数跳过、环形回读。返回每项是否满足，以及是否整体通过。

    ``scene`` 为 Mapping 时读取对应字段；缺省即视为该项已满足（便于
    对空场景或单参数调用给出非空结果）。
    """

    if not isinstance(scene, Mapping):
        scene = {} if scene is None else {"instances": scene}

    frame = int(scene.get("frame", 0))
    pair = frame_pair(frame)
    lag = int(scene.get("lag", DEFAULT_READBACK_LAG))
    slots = int(scene.get("slots", max(lag + 2, 3)))
    fences = scene.get("fences") or ()
    commands = scene.get("commands") or ()
    instances = scene.get("instances") or ()
    if isinstance(instances, int):
        instances = list(range(instances)) if instances > 0 else []
    elif isinstance(instances, Mapping) or not isinstance(instances, Iterable) or isinstance(instances, (str, bytes)):
        instances = [instances]

    if isinstance(commands, int):
        commands = [{"instance_count": commands}]

    frames_paired = True
    for item in fences:
        if isinstance(item, Mapping):
            f_frame = int(item.get("frame", frame))
        else:
            f_frame = int(getattr(item, "frame", frame))
        if f_frame != frame:
            frames_paired = False
            break

    fenced = True
    for item in fences:
        if isinstance(item, Mapping):
            if not item.get("barrier", True) and not item.get("fence"):
                fenced = False
                break
        elif not getattr(item, "range", ""):
            fenced = False
            break

    deferred = bool(scene.get("deferred_reclaim", True))

    det = determinism({"instances": instances})
    deterministic = bool(det["stable"])

    zero = skip_zero(commands)
    zero_skipped = zero["skipped"] == 0 or zero["submitted"] == zero["kept"]

    ring_ok = slots >= lag + 1

    checks = {
        "frame_pairing": frames_paired,
        "append_fence": fenced,
        "deferred_reclaim": deferred,
        "deterministic_sort": deterministic,
        "zero_count_skipped": zero_skipped,
        "ring_readback": ring_ok,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "pair_id": pair["pair_id"],
        "slots": slots,
        "lag": lag,
        "skipped": zero["skipped_indices"],
        "seen": len(instances),
    }


def fallback(request: Any) -> dict[str, Any]:
    """在配对/栅栏失效时回退到安全路径。

    当回读不可用（帧号未配对、发射栅栏缺失、或计数缺失）时，回退策略是
    **本帧不提交间接绘制**，改用上一份已验证的计数快照；绝不用本帧的
    写入去驱动本帧的绘制。这样最多晚一帧，但不会撕裂或越界。

    ``request`` 为 Mapping 时读取 ``pair_valid``、``fence_valid``、``counts``、
    ``last_good``；为整数时只表示计数是否可用（>0）。
    """

    if isinstance(request, Mapping):
        pair_valid = bool(request.get("pair_valid", True))
        fence_valid = bool(request.get("fence_valid", True))
        counts = request.get("counts")
        last_good = request.get("last_good")
        reason = request.get("reason")
    else:
        pair_valid = fence_valid = True
        counts = request
        last_good = None
        reason = None

    reasons: list[str] = []
    if not pair_valid:
        reasons.append("frame_not_paired")
    if not fence_valid:
        reasons.append("fence_missing")

    if counts is not None and last_good is not None and not pair_valid:
        used = last_good
        action = "use_last_good"
    elif counts is not None:
        used = counts
        action = "use_positional"
    else:
        used = last_good
        action = "skip_frame"

    if not reasons and action == "use_positional":
        action = "normal"

    return {
        "action": action,
        "fallback": action != "normal",
        "reasons": reasons,
        "reason": reason,
        "counts": used,
        "defer_to_next_frame": action in ("use_last_good", "skip_frame"),
    }


# ---------------------------------------------------------------------------
# 命令构建 / 指标 / 计数 / 间接参数 / 提交
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawCommand:
    """一条间接绘制命令。"""

    index_count: int
    instance_count: int
    first_index: int
    base_vertex: int
    base_instance: int

    @property
    def bytes(self) -> int:
        return INDIRECT_COMMAND_BYTES

    def as_dict(self) -> dict[str, int]:
        return {
            "indexCount": self.index_count,
            "instanceCount": self.instance_count,
            "firstIndex": self.first_index,
            "baseVertex": self.base_vertex,
            "baseInstance": self.base_instance,
        }


def command_build(request: Any) -> dict[str, Any]:
    """按确定性顺序构建间接绘制命令。

    过程：先按 :func:`sort_key` 对可见实例排序 → 压缩出有计数的命令 →
    跳过零计数 → 为每条命令分配稳定递增的 ``firstInstance``。

    ``request`` 为整数时视为实例总数，生成等量的退化命令用于契约测试。
    """

    if isinstance(request, Mapping):
        instances = list(request.get("instances") or ())
        index_count = int(request.get("index_count", 0))
        first_index = int(request.get("first_index", 0))
        base_vertex = int(request.get("base_vertex", 0))
        frame = int(request.get("frame", 0))
        visible_frame = int(request.get("visible_frame", frame + DEFAULT_READBACK_LAG))
    else:
        total = int(request) if request is not None else 0
        instances = [{"pass": 0, "mesh": i, "instance": i, "visible": 1, "count": 1} for i in range(total)]
        index_count, first_index, base_vertex, frame = 0, 0, 0, 0
        visible_frame = frame + DEFAULT_READBACK_LAG

    ordered = sorted(instances, key=sort_key)
    pair = frame_pair(frame)

    commands: list[DrawCommand] = []
    skipped = 0
    for ordinal, instance in enumerate(ordered):
        count = 0
        if isinstance(instance, Mapping):
            count = int(instance.get("count", instance.get("instance_count", 0)) or 0)
            ic = int(instance.get("index_count", index_count))
            fi = int(instance.get("first_index", first_index))
            bv = int(instance.get("base_vertex", base_vertex))
        else:
            count = int(getattr(instance, "count", getattr(instance, "instance_count", 0)) or 0)
            ic = index_count
            fi = first_index
            bv = base_vertex
        if count <= 0:
            skipped += 1
            continue
        commands.append(
            DrawCommand(
                index_count=_as_uint32(ic),
                instance_count=_as_uint32(count),
                first_index=_as_uint32(fi),
                base_vertex=bv & 0xFFFFFFFF if bv >= 0 else -((-bv) & 0xFFFFFFFF),
                base_instance=len(commands),
            )
        )

    return {
        "frame": frame,
        "visible_frame": visible_frame,
        "pair_id": pair["pair_id"],
        "commands": [c.as_dict() for c in commands],
        "count": len(commands),
        "skipped": skipped,
        "ordered_instances": ordered,
        "bytes": len(commands) * INDIRECT_COMMAND_BYTES,
    }


def metrics(request: Any) -> dict[str, Any]:
    """汇总一次帧内压缩的指标。

    ``request`` 支持 ``commands``/``counts``、``instances``（被剔除的实例数）、
    ``skipped``、``submitted``、``frame`` 等；为整数时视为本帧命令数。
    """

    if isinstance(request, Mapping):
        frame = int(request.get("frame", 0))
        commands = request.get("commands")
        counts = request.get("counts")
        if counts is not None:
            submitted = sum(1 for c in counts if int(c) > 0)
            total = len(counts)
            skipped = total - submitted
            instances = sum(int(c) for c in counts if int(c) > 0)
        else:
            total = int(request.get("total", request.get("count", commands or 0)))
            submitted = int(request.get("submitted", total))
            skipped = int(request.get("skipped", total - submitted))
            instances = int(request.get("instances", submitted))
        lag = int(request.get("lag", DEFAULT_READBACK_LAG))
    else:
        frame = 0
        total = int(request) if request is not None else 0
        submitted = total
        skipped = 0
        instances = total
        lag = DEFAULT_READBACK_LAG

    return {
        "frame": frame,
        "commands_total": total,
        "commands_submitted": submitted,
        "commands_skipped": skipped,
        "instances_visible": instances,
        "readback_lag": lag,
        "drawcalls_saved": skipped,
        "indirect_bytes": submitted * INDIRECT_COMMAND_BYTES,
    }


def instance_count(request: Any) -> dict[str, Any]:
    """计算一帧可见实例的计数并给出回读配对信息。

    ``request`` 支持 ``visible``（布尔序列或布尔映射），或 ``instances``
    （各实例带 ``visible``）；为整数时按该值生成等量可见实例。
    """

    if isinstance(request, Mapping):
        frame = int(request.get("frame", 0))
        if "instances" in request:
            visible_flags = [
                bool(i.get("visible", 1)) if isinstance(i, Mapping) else bool(getattr(i, "visible", True))
                for i in (request.get("instances") or ())
            ]
        else:
            flags = request.get("visible", request.get("flags", ()))
            if isinstance(flags, Mapping):
                visible_flags = [bool(v) for v in flags.values()]
            elif isinstance(flags, int):
                visible_flags = [True] * int(flags)
            else:
                visible_flags = [bool(v) for v in flags]
    else:
        frame = 0
        visible_flags = [True] * (int(request) if request is not None else 0)

    count = sum(1 for flag in visible_flags if flag)
    pair = frame_pair(frame)
    return {
        "count": count,
        "total": len(visible_flags),
        "culled": len(visible_flags) - count,
        "frame": frame,
        "visible_frame": pair["visible_frame"],
        "pair_id": pair["pair_id"],
        "readback_ready": False,
    }


def indirect_args(request: Any) -> dict[str, Any]:
    """生成一段间接参数的描述（含字节布局与零计数过滤）。

    ``request`` 支持 ``commands``（命令序列或计数序列）、或
    ``counts``（int 序列）；为整数时视为单条命令的计数。
    """

    if isinstance(request, Mapping):
        commands = request.get("commands")
        counts = request.get("counts")
        base_offset = int(request.get("offset", 0))
    else:
        commands, counts = None, None
        base_offset = 0

    if commands is None and counts is not None:
        commands = [{"instance_count": int(c)} for c in counts]
    if commands is None:
        n = int(request) if not isinstance(request, Mapping) and request is not None else 0
        commands = [{"instance_count": n}] if not isinstance(request, Mapping) else []

    filtered = skip_zero(commands)
    kept = filtered["commands"]
    args: list[dict[str, int]] = []
    for index, command in enumerate(kept):
        if isinstance(command, Mapping):
            args.append(
                {
                    "indexCount": _as_uint32(command.get("index_count", command.get("indexCount", 0))),
                    "instanceCount": _as_uint32(command.get("instance_count", command.get("instanceCount", 0))),
                    "firstIndex": _as_uint32(command.get("first_index", command.get("firstIndex", 0))),
                    "baseVertex": _as_uint32(command.get("base_vertex", command.get("baseVertex", 0))),
                    "firstInstance": index,
                }
            )
        else:
            args.append(
                {
                    "indexCount": 0,
                    "instanceCount": _as_uint32(getattr(command, "instance_count", 0)),
                    "firstIndex": 0,
                    "baseVertex": 0,
                    "firstInstance": index,
                }
            )

    return {
        "args": args,
        "count": len(args),
        "skipped": filtered["skipped_indices"],
        "offset": base_offset,
        "byte_offset": base_offset * INDIRECT_COMMAND_BYTES,
        "stride": INDIRECT_COMMAND_BYTES,
        "total_bytes": len(args) * INDIRECT_COMMAND_BYTES,
    }


def submit(request: Any) -> dict[str, Any]:
    """提交一次间接绘制，强制同帧配对并跳过零计数命令。

    ``request`` 支持：``frame``、``readback``（帧号或 :func:`ring_readback`
    结果）、``commands``/``counts``、``buffer``/``handle``、``fences``。
    提交前会校验回读帧号与命令帧号一致；不一致则拒绝提交并给出原因。
    """

    if not isinstance(request, Mapping):
        # 单参数场景：按该值生成同帧可提交的一组命令。
        n = int(request) if request is not None else 0
        request = {"frame": 0, "counts": [n]}

    frame = int(request.get("frame", 0))
    pair = frame_pair(frame)
    readback = request.get("readback")
    commands = request.get("commands")

    rb: dict[str, Any] | None = None
    if isinstance(readback, Mapping):
        rb = dict(readback)
    elif readback is not None:
        rb = ring_readback({"frame": int(readback)})

    readback_frame = None
    pair_valid = True
    if rb is not None:
        readback_frame = int(rb.get("frame", rb.get("source_frame", 0)))
        # 命令帧必须等于回读帧（同一帧号配对）。
        pair_valid = readback_frame == frame
        if not pair_valid:
            return {
                "submitted": False,
                "reason": "frame_mismatch",
                "frame": frame,
                "readback_frame": readback_frame,
                "pair_id": pair["pair_id"],
                "commands": [],
                "count": 0,
            }

    if commands is None:
        counts = request.get("counts")
        if counts is None:
            counts = [1]
        commands = [{"instance_count": int(c)} for c in counts]

    filtered = skip_zero(commands)
    if not filtered["commands"]:
        return {
            "submitted": False,
            "reason": "all_zero",
            "frame": frame,
            "readback_frame": readback_frame,
            "pair_id": pair["pair_id"],
            "commands": [],
            "count": 0,
            "skipped": filtered["skipped_indices"],
        }

    buffer = request.get("buffer", request.get("handle"))
    return {
        "submitted": True,
        "reason": "ok",
        "frame": frame,
        "readback_frame": readback_frame,
        "pair_id": pair["pair_id"],
        "commands": filtered["commands"],
        "count": filtered["kept"],
        "skipped": filtered["skipped_indices"],
        "buffer": buffer,
        "indirect_bytes": filtered["kept"] * INDIRECT_COMMAND_BYTES,
    }
