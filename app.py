import json
import base64
import re
import time
import pprint
import traceback
import threading

import requests
import pandas as pd
from pypinyin import lazy_pinyin
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# ── 鉴权配置 ──────────────────────────────────────────────
APPID = "com.noah.pangu.rl"
TOKEN = "xxxx"
BASE_URL = "https://roma.huawei.com/csb/rest/saas/ei/eiWizard"
COMMON_PARAMS = {
    "appid": APPID,
    "vendor": "HEC",
    "region": "cn-southwest-2",
}
HEADERS = {
    "content-type": "application/json;charset=UTF-8",
    "csb-token": TOKEN,
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
df = pd.read_excel('算法卡池先导用卡分配.xlsx', sheet_name='能力项用卡名单')
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
              status_field=None, status_value=None,
              duration_field=None, extra_fields=None):
    """
    通用汇总函数，适配训练作业、开发环境、推理服务。

    extra_fields: list of (src_key, dst_key) 额外字段映射
    """
    leader_data = {}
    spec_gpu = {}
    extra_fields = extra_fields or []

    for item in items:
        # 状态过滤（可选）
        if status_field and status_value:
            if item.get(status_field) != status_value:
                continue

        user_id = item.get("userId") or ""
        spec_name = item.get(spec_field) or "未知规格"
        gpu_num = item.get(gpu_field) or 0
        item_name = item.get(name_field) or ""

        duration = 0
        if duration_field:
            dur_str = item.get(duration_field, "0:0:0") or "0:0:0"
            duration = int(dur_str.split(':')[0]) if ':' in dur_str else 0

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
def _get(url, extra_params, filter_dict):
    params = {**COMMON_PARAMS, **extra_params}
    encoded = base64.b64encode(json.dumps(filter_dict, ensure_ascii=False).encode()).decode()
    params["params"] = encoded
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        print(f"[HTTP {r.status_code}] {url}")
    except Exception as e:
        print(f"请求异常 {url}: {e}")
        traceback.print_exc()
    return None


# ── 训练作业 ──────────────────────────────────────────────
def fetch_train_data():
    print("获取训练作业数据...")
    data = _get(
        f"{BASE_URL}/train/job/list",
        {"trainApiVersion": "V2"},
        {
            "pageSize": "500", "pageIndex": "0", "status": "8",
            "searchName": "", "filterParam": [{"key": "", "value": ""}],
            "tagIds": [""],
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


# ── 开发环境（Notebook）────────────────────────────────────
# TODO: 确认实际接口路径和返回字段后按需调整
def fetch_devenv_data():
    print("获取开发环境数据...")
    data = _get(
        f"{BASE_URL}/notebook/list",
        {},
        {
            "pageSize": "500", "pageIndex": "0",
            "status": "RUNNING",          # 运行中
            "searchName": "",
        },
    )
    if data is None:
        return None, None
    return aggregate(
        data.get("instances", []),        # TODO: 确认列表字段名
        gpu_field="workingGpuNum",
        name_field="name",
        spec_field="flavor",              # TODO: 确认规格字段名
        status_field="status",
        status_value="RUNNING",
        duration_field="duration",
    )


# ── 推理服务 ──────────────────────────────────────────────
# TODO: 确认实际接口路径和返回字段后按需调整
def fetch_inference_data():
    print("获取推理服务数据...")
    data = _get(
        f"{BASE_URL}/inference/service/list",
        {},
        {
            "pageSize": "500", "pageIndex": "0",
            "status": "running",
            "searchName": "",
        },
    )
    if data is None:
        return None, None
    return aggregate(
        data.get("services", []),          # TODO: 确认列表字段名
        gpu_field="gpuNum",                # TODO: 确认 GPU 字段名
        name_field="name",
        spec_field="specName",
        status_field="status",
        status_value="running",
        duration_field="duration",
        extra_fields=[("instanceCount", "instance_count")],
    )


# ── 缓存刷新 ──────────────────────────────────────────────
def refresh_cache():
    """获取三类数据并更新缓存。调用方需自己持有 _cache_lock。"""
    fetchers = {
        "train":     fetch_train_data,
        "devenv":    fetch_devenv_data,
        "inference": fetch_inference_data,
    }
    updated = False
    for key, fn in fetchers.items():
        user_data, spec_data = fn()
        if user_data is not None:
            _cache[key]["user_data"] = user_data
            _cache[key]["spec_data"] = spec_data
            updated = True
        else:
            print(f"[{key}] 获取失败，保留旧缓存")
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
    app.run(host='0.0.0.0', port=5063, debug=False)
