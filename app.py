import json
import base64
import re
import time
import traceback
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from pypinyin import lazy_pinyin
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# ── 鉴权配置 ──────────────────────────────────────────────
ENDPOINT      = "https://roma.huawei.com"   # 替换为实际 Endpoint
APPID         = "com.noah.pangu.rl"
API_VERSION   = "v1"                        # demanager 接口 version 参数
VENDOR        = "HEC"
REGION        = "cn-southwest-2"
csb_token = "xxxx"                      # Authorization header 值
X_HW_ID       = "xxxx"                      # X-HW-ID header 值
X_HW_APPKEY   = "xxxx"                      # X-HW-APPKEY header 值

HEADERS = {
    "content-Type":  "application/json",
    "csb-token": csb_token,
    "X-HW-ID":       X_HW_ID,
    "X-HW-APPKEY":   X_HW_APPKEY,
}

# ── 缓存配置 ──────────────────────────────────────────────
CACHE_EXPIRE = 300  # 5 分钟

_cache_lock = threading.Lock()
_cache = {
    "train":     {"user_data": {}, "spec_data": {}},
    "devenv":    {"user_data": {}, "spec_data": {}},
    "inference": {"user_data": {}, "spec_data": {}},
    "last_update": 0,
}

# ── 花名册解析 ────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
_EXCEL_PATH = _BASE_DIR / '算法卡池先导用卡分配.xlsx'
try:
    df = pd.read_excel(_EXCEL_PATH, sheet_name='能力项用卡名单')
    print(f"花名册加载成功：{_EXCEL_PATH}，共 {len(df)} 行")
except FileNotFoundError:
    raise SystemExit(f"[ERROR] 找不到花名册文件：{_EXCEL_PATH}")
capability_columns = df.columns[1:-2].tolist()

usr_dict = {}       # key → leader
usr_name_dict = {}  # key → 用户全名


def get_first_letter(text):
    text = str(text).strip()
    if not text:
        return ''
    c = text[0]
    return lazy_pinyin(c)[0][0].lower() if '\u4e00' <= c <= '\u9fff' else c.lower()


def extract_id(text):
    m = re.search(r'\d+', str(text))
    return m.group() if m else ''


for col in capability_columns:
    leader = str(df[col].iloc[0]).strip() if pd.notna(df[col].iloc[0]) else ''
    for member in df[col].iloc[1:]:
        if pd.notna(member) and str(member).strip() not in ('nan', 'sum', ''):
            s = str(member).strip()
            mid = extract_id(s)
            key = f'{get_first_letter(s)}{mid}' if mid else get_first_letter(s)
            usr_dict[key] = leader
            usr_name_dict[key] = s
            if mid:
                usr_dict[mid] = leader
                usr_name_dict[mid] = s


# ── 通用：用户信息查找 ────────────────────────────────────
def resolve_user(user_id):
    """返回 (user_name, leader_name)，找不到时降级返回 user_id 本身。"""
    stripped = user_id.lstrip(user_id[0]) if user_id and user_id[0].isalpha() else None
    user_name = usr_name_dict.get(user_id) or usr_name_dict.get(stripped) or user_id
    leader_name = usr_dict.get(user_id) or usr_dict.get(stripped) or user_name
    return user_name, leader_name


# ── 通用：按 leader→member→tasks 汇总 ────────────────────
def aggregate(items, *, gpu_field, name_field, spec_field,
              user_field="userId",
              status_field=None, status_value=None,
              region_field=None, region_value=None,
              duration_field=None, extra_fields=None):
    """
    通用汇总函数，适配训练作业、开发环境、推理服务。

    user_field:     记录用户 ID 的字段名，默认 "userId"
    region_field/region_value: 可选的 region 过滤
    duration_field: 值可为 "HH:MM:SS" 字符串，或毫秒时间戳（createTime）
    extra_fields:   list of (src_key, dst_key) 额外字段映射
    """
    leader_data = {}
    spec_gpu = {}
    extra_fields = extra_fields or []

    now_ms = time.time() * 1000

    for item in items:
        # 状态过滤
        if status_field and status_value:
            if item.get(status_field) != status_value:
                continue
        # region 过滤
        if region_field and region_value:
            if item.get(region_field) != region_value:
                continue

        user_id = item.get(user_field) or ""
        spec_name = item.get(spec_field) or "未知规格"
        gpu_num = int(float(item.get(gpu_field) or 0))  # 兼容 "4" / "8.0"
        item_name = item.get(name_field) or ""

        duration = 0
        if duration_field:
            raw = item.get(duration_field)
            if raw:
                raw_str = str(raw)
                if ':' in raw_str:
                    # "HH:MM:SS" 格式
                    duration = int(raw_str.split(':')[0])
                elif raw_str.isdigit() and int(raw_str) > 1_000_000_000_000:
                    # 毫秒时间戳（createTime）→ 计算已运行小时数
                    duration = int((now_ms - int(raw_str)) / 3_600_000)

        user_name, leader_name = resolve_user(user_id)

        task_json = {
            'user':      user_name,
            'gpu_num':   gpu_num,
            'duration':  duration,
            'task_name': item_name,
            'spec_name': spec_name,
        }
        for src, dst in extra_fields:
            task_json[dst] = item.get(src)

        # leader 层
        if leader_name not in leader_data:
            leader_data[leader_name] = {
                'gpu_num': 0, 'task_count': 0,
                'total_duration': 0, 'max_duration': 0,
                'members': {},
            }
        ld = leader_data[leader_name]
        ld['gpu_num']       += gpu_num
        ld['task_count']    += 1
        ld['total_duration'] += duration

        # member 层
        if user_name not in ld['members']:
            ld['members'][user_name] = {
                'gpu_num': 0, 'task_count': 0,
                'total_duration': 0, 'max_duration': 0,
                'tasks': [],
            }
        md = ld['members'][user_name]
        md['gpu_num']       += gpu_num
        md['task_count']    += 1
        md['total_duration'] += duration
        md['max_duration']   = max(md['max_duration'], duration)
        md['tasks'].append(task_json)

        # 资源池汇总
        spec_gpu[spec_name] = spec_gpu.get(spec_name, 0) + gpu_num

    # leader 最长 = 成员最长之最大
    for ld in leader_data.values():
        ld['max_duration'] = max(
            (m['max_duration'] for m in ld['members'].values()), default=0
        )

    return leader_data, spec_gpu


# ── API 请求基础函数 ──────────────────────────────────────
def _b64(obj):
    """将 dict 序列化后 base64 编码，供 params 字段使用。"""
    return base64.b64encode(json.dumps(obj, ensure_ascii=False).encode()).decode()


def _get(url, params, timeout=15):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        print(f"[GET {r.status_code}] {url}")
    except Exception as e:
        print(f"GET 请求异常 {url}: {e}")
        traceback.print_exc()
    return None


def _post(url, params, body, timeout=15):
    try:
        r = requests.post(url, params=params, json=body, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        print(f"[POST {r.status_code}] {url}")
    except Exception as e:
        print(f"POST 请求异常 {url}: {e}")
        traceback.print_exc()
    return None


# ── 开发环境 ──────────────────────────────────────────────
def fetch_devenv_data():
    """POST /csb/roma-aistudio/demanager/list"""
    print("获取开发环境数据...")
    data = _post(
        f"{ENDPOINT}/csb/roma-aistudio/demanager/list",
        params={"appid": APPID, "version": API_VERSION},
        body={
            "vendor":   VENDOR,
            "region":   REGION,
            "deType":   "",     # 不过滤类型，返回全部
            "pageNum":  1,
            "pageSize": 500,
        },
    )
    if data is None:
        return None, None
    items = data.get("notebooks") or []
    return aggregate(
        items,
        user_field="creator",
        gpu_field="npuNum",
        name_field="name",
        spec_field="flavor",
        status_field="status",
        status_value="RUNNING",
        region_field="region",
        region_value=REGION,
        duration_field="createTime",
    )


# ── 训练作业 ──────────────────────────────────────────────
def fetch_train_data():
    """GET /csb/roma-aistudio/train/job/list"""
    print("获取训练作业数据...")
    data = _get(
        f"{ENDPOINT}/csb/roma-aistudio/train/job/list",
        params={
            "appid":            APPID,
            "trainApiVersion":  "V2",
            "jobType":          "",
            "region":           REGION,
            "params": _b64({
                "pageSize":  "500",
                "pageIndex": "0",
#                "status":    "8",
            }),
        },
    )
    if data is None:
        return None, None
    return aggregate(
        data.get("trainJobs", []),
        gpu_field="workingGpuNum",
        name_field="name",
        spec_field="specName",
        status_field="statusCode",
        status_value="8",
        duration_field="duration",
    )


# ── 推理服务（v1 + v2 合并）────────────────────────────────
def _fetch_inference_v1():
    """GET /csb/roma-aistudio/infer/real-time/service/list"""
    return _get(
        f"{ENDPOINT}/csb/roma-aistudio/infer/real-time/service/list",
        params={
            "appid":     APPID,
            "infertype": "real-time",
            "params": _b64({
                "pageSize":    500,
                "pageIndex":   1,
                "filterParam": [{"key": "name", "value": ""}],
            }),
        },
    )


def _fetch_inference_v2():
    """GET /csb/roma-aistudio/infer/real-time/service/v2/list"""
    return _get(
        f"{ENDPOINT}/csb/roma-aistudio/infer/real-time/service/v2/list",
        params={
            "appid":     APPID,
            "infertype": "real-time",
            "vendor":    VENDOR,
            "params": _b64({
                "pageSize":    500,
                "pageIndex":   1,
                "filterParam": [{"key": "name", "value": ""}],
            }),
        },
    )


def _merge_aggregations(a, b):
    """将两次 aggregate() 结果 (leader_data, spec_gpu) 合并为一份。"""
    if not a[0]:
        return b
    if not b[0]:
        return a
    ld_a, sp_a = a
    ld_b, sp_b = b
    merged_ld = {k: v for k, v in ld_a.items()}
    for leader, ld in ld_b.items():
        if leader not in merged_ld:
            merged_ld[leader] = ld
        else:
            merged_ld[leader]['gpu_num']        += ld['gpu_num']
            merged_ld[leader]['task_count']     += ld['task_count']
            merged_ld[leader]['total_duration'] += ld['total_duration']
            merged_ld[leader]['max_duration']    = max(
                merged_ld[leader]['max_duration'], ld['max_duration'])
            for member, md in ld['members'].items():
                if member not in merged_ld[leader]['members']:
                    merged_ld[leader]['members'][member] = md
                else:
                    m = merged_ld[leader]['members'][member]
                    m['gpu_num']        += md['gpu_num']
                    m['task_count']     += md['task_count']
                    m['total_duration'] += md['total_duration']
                    m['max_duration']    = max(m['max_duration'], md['max_duration'])
                    m['tasks'].extend(md['tasks'])
    merged_sp = {k: v for k, v in sp_a.items()}
    for k, v in sp_b.items():
        merged_sp[k] = merged_sp.get(k, 0) + v
    return merged_ld, merged_sp


def fetch_inference_data():
    """分别处理 v1 / v2 推理服务（字段不同），再合并结果"""
    print("获取推理服务数据...")

    # ── v1 ─────────────────────────────────────────────────
    result_v1 = (None, None)
    data_v1 = _fetch_inference_v1()
    if data_v1:
        items_v1 = (data_v1.get("services")
                    or data_v1.get("modelServiceList")
                    or data_v1.get("serviceList") or [])
        if items_v1:
            result_v1 = aggregate(
                items_v1,
                user_field="creator",
                gpu_field="xpuNum",
                name_field="name",
                spec_field="inferType",
                status_field="status",
                status_value="running",
                region_field="region",
                region_value=REGION,
                duration_field="publishTime",
            )

    # ── v2 ─────────────────────────────────────────────────
    result_v2 = (None, None)
    data_v2 = _fetch_inference_v2()
    if data_v2:
        items_v2 = (data_v2.get("services")
                    or data_v2.get("modelServiceList")
                    or data_v2.get("serviceList") or [])
        if items_v2:
            result_v2 = aggregate(
                items_v2,
                user_field="creator",
                gpu_field="xpuNum",
                name_field="name",
                spec_field="inferType",
                status_field="status",
                status_value="running",
                region_field="region",
                region_value=REGION,
                duration_field="publishTime",
            )

    merged = _merge_aggregations(result_v1, result_v2)
    if not merged[0]:
        return None, None
    return merged


# ── 缓存刷新 ──────────────────────────────────────────────
def refresh_cache():
    """并行拉取三类数据并更新缓存。调用方需自己持有 _cache_lock。"""
    fetchers = {
        "train":     fetch_train_data,
        "devenv":    fetch_devenv_data,
        "inference": fetch_inference_data,
    }
    updated = False
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): key for key, fn in fetchers.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                user_data, spec_data = future.result()
                if user_data is not None:
                    _cache[key]["user_data"] = user_data
                    _cache[key]["spec_data"] = spec_data
                    updated = True
                else:
                    print(f"[{key}] 获取失败，保留旧缓存")
            except Exception as e:
                print(f"[{key}] 异常：{e}")
    if updated:
        _cache["last_update"] = time.time()


def get_cached_data():
    """返回三类缓存数据，必要时刷新。线程安全。"""
    with _cache_lock:
        age = time.time() - _cache["last_update"]
        if age > CACHE_EXPIRE or not any(_cache[k]["user_data"] for k in ("train", "devenv", "inference")):
            print(f"缓存过期（{age:.0f}s），重新拉取...")
            refresh_cache()
        else:
            print(f"使用缓存（{age:.0f}s / {CACHE_EXPIRE}s）")
        return {
            "train":     dict(_cache["train"]),
            "devenv":    dict(_cache["devenv"]),
            "inference": dict(_cache["inference"]),
            "last_update": _cache["last_update"],
        }


# ── 路由 ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index_multi.html')


@app.route('/data')
def get_data():
    d = get_cached_data()
    return jsonify({
        "train":     d["train"],
        "devenv":    d["devenv"],
        "inference": d["inference"],
        "update_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(d["last_update"])),
    })


if __name__ == '__main__':
    import socket
    PORT = 5063
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', PORT)) == 0:
            raise SystemExit(f"[ERROR] 端口 {PORT} 已被占用，请先停止旧进程（lsof -i :{PORT}）")
    app.run(host='0.0.0.0', port=PORT, debug=False)
