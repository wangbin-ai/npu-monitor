import json
import base64
import re
import time
import traceback
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib3
import yaml
import requests
from pypinyin import lazy_pinyin

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from flask import Flask, render_template, jsonify, request


def _load_config():
    """解析 --config / --roster 路径并校验文件存在，返回 (auth_cfg, roster_path)。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=None)
    parser.add_argument('--roster', default=None)
    args, _ = parser.parse_known_args()

    if not args.config:
        raise SystemExit(
            "[ERROR] 请通过 --config <path> 指定鉴权配置文件\n"
            "  例如：python app.py --config config.yaml --roster roster.yaml"
        )
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"[ERROR] 鉴权配置文件不存在：{config_path.resolve()}")

    if not args.roster:
        raise SystemExit("[ERROR] 请通过 --roster <path> 指定花名册 YAML 文件")
    roster_path = Path(args.roster)
    if not roster_path.exists():
        raise SystemExit(f"[ERROR] 花名册文件不存在：{roster_path.resolve()}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg, roster_path


_cfg, _roster_path = _load_config()

# ── ROMA 平台配置（从 YAML roma: 节加载）─────────────────────
_roma = _cfg['roma']
ENDPOINT    = _roma['endpoint']
APPID       = _roma['appid']
API_VERSION = _roma['api_version']
VENDOR      = _roma['vendor']
REGION      = _roma['region']

HEADERS = {
    "content-Type": "application/json",
    "csb-token":    _roma['csb_token'],
    "X-HW-ID":      _roma['x_hw_id'],
    "X-HW-APPKEY":  _roma['x_hw_appkey'],
}

# ── MA (ModelArts) 平台配置（可选）────────────────────────────
_ma = _cfg.get('ma') or {}
if _ma and all(_ma.get(k) for k in ('csb_endpoint', 'project_id', 'cloud_provider',
                                     'resource_type', 'region', 'cloud_project_id')):
    MA_BASE = (
        f"{_ma['csb_endpoint']}/csb/projects/{_ma['project_id']}"
        f"/vendors/{_ma['cloud_provider']}"
        f"/resourcetypes/{_ma['resource_type']}"
        f"/regions/{_ma['region']}"
        f"/projects/{_ma['cloud_project_id']}"
    )
    MA_PROJECT = _ma['cloud_project_id']
    MA_HEADERS = {
        "Accept":        "application/json;charset=utf-8",
        "Authorization": _ma.get('Authorization', ''),
        "Content-Type":  "application/json",
    }
    print(f"[MA] 配置加载成功，base: {MA_BASE}")
else:
    MA_BASE = MA_PROJECT = MA_HEADERS = None
    print("[MA] 未配置或配置不完整，跳过 MA 查询")


def _fetch_ma_token():
    """调用 IAM 接口获取 MA Authorization token，返回 token 字符串或 None。"""
    token_cfg = _ma.get('token') or {}
    endpoint   = token_cfg.get('endpoint', 'https://iam.his-op.huawei.com/iam/auth/token')
    secret     = token_cfg.get('secret', '')
    enterprise = token_cfg.get('enterprise', '')
    account    = token_cfg.get('account', '')
    project    = token_cfg.get('project', '')
    if not (secret and enterprise and account and project):
        return None
    body = {
        "data": {
            "type": "jwt-token",
            "attributes": {
                "account":    account,
                "secret":     secret,
                "project":    project,
                "enterprise": enterprise,
            }
        }
    }
    try:
        r = requests.post(endpoint, json=body, timeout=15,
                          verify=False, proxies={"http": None, "https": None})
        if r.status_code in (200, 201):
            resp = r.json()
            token = (resp.get('access_token')
                     or (resp.get('data') or {}).get('access_token'))
            if token:
                return token
            print(f"[MA Token] 响应中未找到 access_token，返回体：{str(resp)[:200]}")
        else:
            print(f"[MA Token] 获取失败 {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[MA Token] 请求异常: {e}")
    return None


def _ma_token_refresher():
    """后台守护线程：启动时立即获取 token，之后每隔 23.5 小时刷新。"""
    while True:
        token = _fetch_ma_token()
        if token and MA_HEADERS is not None:
            MA_HEADERS['Authorization'] = token
            print(f"[MA Token] 刷新成功（前20字符）：{token[:20]}…")
        else:
            print("[MA Token] 刷新失败，保留当前 token")
        time.sleep(23.5 * 3600)


_token_cfg = _ma.get('token') or {}
if MA_BASE and all(_token_cfg.get(k) for k in ('secret', 'enterprise', 'account', 'project')):
    _t = threading.Thread(target=_ma_token_refresher, daemon=True, name="ma-token-refresher")
    _t.start()

app = Flask(__name__)
# ── 缓存配置 ──────────────────────────────────────────────
CACHE_EXPIRE = 60  # 5 分钟

_cache_lock = threading.Lock()
_cache = {
    "train":     {"user_data": {}, "spec_data": {}},
    "devenv":    {"user_data": {}, "spec_data": {}},
    "inference": {"user_data": {}, "spec_data": {}},
    "ma_devenv": {"user_data": {}, "spec_data": {}},
    "ma_train":  {"user_data": {}, "spec_data": {}},
    "last_update": 0,
}

# ── 花名册 ────────────────────────────────────────────────
usr_dict = {}       # key (lowercase) → leader
usr_name_dict = {}  # key (lowercase) → 用户全名
quota_dict = {}     # leader_name → 配额NPU卡数

_UNKNOWN_LEADER = "__unknown__"  # 哨兵：不在花名册的用户统一归入此组


def get_first_letter(text):
    text = str(text).strip()
    if not text:
        return ''
    c = text[0]
    return lazy_pinyin(c)[0][0].lower() if '\u4e00' <= c <= '\u9fff' else c.lower()


def _parse_member_key(s):
    """将 Excel 成员单元格解析为 (key, mid)，key 与 API 返回的 user_id 格式匹配。

    API user_id 格式统一为：姓名拼音首字母 + 工号/ID
      "Wang Tingkuo 84442956"  → ("w84442956",  "84442956")  英文拼音名+纯数字工号
      "Zhang Zhi 84413741"     → ("z84413741",  "84413741")
      "范诗卿 00934895"        → ("f00934895",  "00934895")  中文名+空格+纯数字工号
      "李媚 wx1209009"         → ("lwx1209009", "wx1209009") 中文名+空格+字母数字ID
                                  （API返回 l+wx1209009，l为李的拼音首字母）
      "w00910350"              → ("w00910350",  "00910350")  纯ID（字母前缀）
      "张某某84434546"         → ("z84434546",  "84434546")  旧格式：中文名直连数字
    """
    parts = s.split()

    if len(parts) >= 2:
        id_part = parts[-1]
        name_part = ' '.join(parts[:-1])
        # 无论工号是纯数字还是字母数字混合，API 格式均为：姓名拼音首字母 + 工号
        first = get_first_letter(name_part)
        key = (first + id_part).lower()
        mid = id_part.lower()   # 去掉首字母后的部分，用于 resolve_user 的 stripped 查找
        return key, mid

    # 单 token（无空格）
    if s and '\u4e00' <= s[0] <= '\u9fff':
        # 中文打头：提取全部数字（兼容"张某某84434546"旧格式）
        digits = re.sub(r'[^\d]', '', s)
        first = get_first_letter(s)
        return (first + digits, digits) if digits else (first, '')
    elif s and s[0].isalpha():
        # 字母打头 ID（如 w00910350）
        k = s.lower()
        return k, k[1:]
    else:
        k = s.lower()
        return k, k


def _store(key, name, leader):
    """将 key 统一转为小写后写入字典，跳过空 key。"""
    k = key.strip().lower()
    if not k:
        return
    usr_name_dict[k] = name
    if leader:
        usr_dict[k] = leader


def _load_roster(path):
    """从 YAML 花名册文件填充 usr_dict / usr_name_dict / quota_dict。"""
    with open(path) as f:
        data = yaml.safe_load(f)
    for group in data.get('groups', []):
        leader = str(group.get('leader', '')).strip()
        if not leader:
            continue
        try:
            q = int(float(group.get('quota', 0) or 0))
        except (ValueError, TypeError):
            q = 0
        if q > 0:
            quota_dict[leader] = q

        lkey, lmid = _parse_member_key(leader)
        if lkey:
            _store(lkey, leader, leader)
        if lmid and lmid != lkey:
            _store(lmid, leader, leader)

        for member in group.get('members', []):
            s = str(member).strip()
            if not s:
                continue
            key, mid = _parse_member_key(s)
            _store(key, s, leader)
            if mid and mid != key:
                _store(mid, s, leader)


_load_roster(_roster_path)
print(f"[花名册] 共加载 {len(usr_dict)} 个用户ID → 组长映射，"
      f"示例：{list(usr_dict.items())[:5]}")


# ── 通用：用户信息查找 ────────────────────────────────────
def resolve_user(user_id):
    """返回 (user_name, leader_name)，找不到时降级返回 user_id 本身。
    所有 key 统一转小写匹配，避免大小写不一致导致漏查。
    """
    if not user_id:
        return user_id or '', user_id or ''

    uid = user_id.strip().lower()
    # 尝试完整 ID 和去掉首字母后的 ID 两种形式
    candidates = [uid]
    if uid and uid[0].isalpha():
        candidates.append(uid[1:])   # 去掉一个前缀字母

    user_name = None
    leader_name = None
    for key in candidates:
        if not key:
            continue
        if user_name is None and key in usr_name_dict:
            user_name = usr_name_dict[key]
        if leader_name is None and key in usr_dict:
            leader_name = usr_dict[key]
        if user_name and leader_name:
            break

    user_name = user_name or user_id          # 找不到全名时用原始 ID
    leader_name = leader_name or _UNKNOWN_LEADER  # 不在花名册→归入非白名单组
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
        # 状态过滤（status_value 可为单值或 set/list）
        if status_field and status_value is not None:
            sv = item.get(status_field)  
            if isinstance(status_value, (set, list, tuple)):
                if sv not in status_value:
                    continue
            else:
                if sv != status_value:
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
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout,
                        verify=False, proxies={"http": None, "https": None})
        if r.status_code == 200:
            return r.json()
        print(f"[GET {r.status_code}] {url}")
    except Exception as e:
        print(f"GET 请求异常 {url}: {e}")
        traceback.print_exc()
    return None


def _post(url, params, body, timeout=15):
    try:
        r = requests.post(url, params=params, json=body, headers=HEADERS, timeout=timeout,
                         verify=False, proxies={"http": None, "https": None})
        if r.status_code == 200:
            return r.json()
        print(f"[POST {r.status_code}] {url}")
    except Exception as e:
        print(f"POST 请求异常 {url}: {e}")
        traceback.print_exc()
    return None


def _ma_get(path, params=None, timeout=15):
    """MA 平台 GET 请求：{MA_BASE}{path}，使用 MA 独立鉴权头。"""
    if not MA_BASE:
        return None
    url = MA_BASE + path
    try:
        r = requests.get(url, params=params or {}, headers=MA_HEADERS, timeout=timeout,
                         verify=False, proxies={"http": None, "https": None})
        if r.status_code == 200:
            return r.json()
        print(f"[MA GET {r.status_code}] {url}")
    except Exception as e:
        print(f"MA GET 请求异常 {url}: {e}")
        traceback.print_exc()
    return None


def _ma_post(path, json_body=None, timeout=15):
    """MA 平台 POST 请求：{MA_BASE}{path}，使用 MA 独立鉴权头。"""
    if not MA_BASE:
        return None
    url = MA_BASE + path
    try:
        r = requests.post(url, json=json_body or {}, headers=MA_HEADERS, timeout=timeout,
                          verify=False, proxies={"http": None, "https": None})
        if r.status_code == 200:
            return r.json()
        print(f"[MA POST {r.status_code}] {url}")
        print(f"  {r.text[:200]}")
    except Exception as e:
        print(f"MA POST 请求异常 {url}: {e}")
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
            "deType":   "Notebook",     # 不过滤类型，返回全部
            "pageNum":  1,
            "pageSize": 10000,
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
    all_jobs = []
    for status in ("26", "7", "8", "24"):
        data = _get(
            f"{ENDPOINT}/csb/roma-aistudio/train/job/list",
            params={
                "appid": APPID,
                "trainApiVersion": "V2",
                "jobType": "",
                "region": REGION,
                "params": _b64({
                    "pageSize": 500,
                    "pageIndex": 1,
                    "status": status,
                }),
            },
        )
        if data is None:
            return None, None
        all_jobs.extend(data.get("trainJobs", []))

    return aggregate(
        all_jobs,
        gpu_field="workingGpuNum",
        name_field="name",
        spec_field="specName",
        status_field="statusCode",
        status_value={"26", "7", "8", "24"},  # 运行中(8)、等待资源(6)、初始化(7)
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
                "pageSize":    10000,
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
                "pageSize":    10000,
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


# ── MA：Notebook 列表（ListAllNotebooks）────────────────────
def fetch_ma_devenv_data():
    """GET /v1/{project_id}/notebooks/all?status=RUNNING，自动翻页。

    响应字段映射：
      user_id      → 归属用户
      name         → notebook 名称
      flavor       → 规格（NPU 型号/规格串，count 不在 API 中故设为 0）
      status       → 过滤 RUNNING
      lease.create_at (ms) → 计算已运行小时数
    """
    if not MA_BASE:
        return None, None
    print("获取 MA Notebook 数据...")
    all_items = []
    offset, limit = 0, 50
    base_params = {
        "feature":     "NOTEBOOK",   # 仅查计费规格，排除免费 CodeLab
        "status":      "RUNNING",
        "limit":       limit,
        "workspaceId": _ma.get("workspace_id", "0"),
    }
    while True:
        data = _ma_get(
            f"/v1/{MA_PROJECT}/notebooks/all",
            params={**base_params, "offset": offset},
        )
        if data is None:
            return None, None
        items = data.get("data") or []
        all_items.extend(items)
        if not items or len(all_items) >= (data.get("total") or 0):
            break
        offset += limit

    # 展开字段：user.name 格式为 "{首字母}{工号}"，去前缀得工号用于花名册匹配
    for item in all_items:
        item["_create_at"] = (item.get("lease") or {}).get("create_at")
        m = re.search(r'\.npu\.(\d+)', item.get("flavor", ""))
        item["_npu_num"] = int(m.group(1)) if m else 0
        raw_name = (item.get("user") or {}).get("name", "")
        item["_user_key"] = raw_name[1:] if raw_name else ""

    return aggregate(
        all_items,
        user_field="_user_key",
        gpu_field="_npu_num",
        name_field="name",
        spec_field="flavor",
        status_field="status",
        status_value="RUNNING",
        duration_field="_create_at",
    )


# ── MA：训练实验列表（ListTrainingExperiments）───────────────
def fetch_ma_train_data():
    """POST /v2/{project_id}/training-job-searches，过滤 Running 状态，自动翻页。

    响应字段映射：
      metadata.user_name   → 归属用户（格式与 Notebook 一致，去首字母前缀）
      metadata.name        → 作业名称
      status.duration (ms) → 已运行时长
      spec.resource        → NPU 卡数 = node_count × flavor_detail.flavor_info.npu.unit_num
    """
    if not MA_BASE:
        return None, None
    print("获取 MA 训练作业数据...")
    all_items = []
    offset, limit = 0, 50
    body_base = {
        "sort_by":     "create_time",
        "order":       "desc",
        "workspace_id": _ma.get("workspace_id", "0"),
        "filters":     [{"key": "phase", "operator": "in", "value": ["Running", "Pending"]}],
    }
    while True:
        data = _ma_post(
            f"/v2/{MA_PROJECT}/training-job-searches",
            json_body={**body_base, "offset": offset, "limit": limit},
        )
        if data is None:
            return None, None
        items = data.get("items") or []
        all_items.extend(items)
        if not items or len(all_items) >= (data.get("total") or 0):
            break
        offset += limit

    now_ms = int(time.time() * 1000)
    for item in all_items:
        meta     = item.get("metadata") or {}
        status   = item.get("status")   or {}
        resource = (item.get("spec") or {}).get("resource") or {}

        raw_name = meta.get("user_name", "")
        item["_user_key"] = raw_name[1:] if raw_name else ""
        item["_name"]     = meta.get("name", "")

        pool_info    = resource.get("pool_info") or {}
        npu_per_node = int(pool_info.get("accelerator_num") or 0)
        node_count   = int(resource.get("node_count") or 1)
        item["_npu_num"] = npu_per_node * node_count

        # status.duration 单位毫秒，转为伪 create_at 供 aggregate 计算小时数
        duration_ms      = int(status.get("duration") or 0)
        item["_create_at"] = now_ms - duration_ms

        sec = status.get("secondary_phase", "")
        status_prefix = {"Creating": "[创建中] ", "Queuing": "[排队中] "}.get(sec, "")
        item["_flavor"] = status_prefix + (pool_info.get("pool_resource_flavor")
                                           or resource.get("flavor_id") or "")

    return aggregate(
        all_items,
        user_field="_user_key",
        gpu_field="_npu_num",
        name_field="_name",
        spec_field="_flavor",
        status_field=None,
        status_value=None,
        duration_field="_create_at",
    )


# ── 缓存刷新 ──────────────────────────────────────────────
def refresh_cache():
    """并行拉取所有平台数据并更新缓存。调用方需自己持有 _cache_lock。"""
    fetchers = {
        "train":     fetch_train_data,
        "devenv":    fetch_devenv_data,
        "inference": fetch_inference_data,
    }
    if MA_BASE:
        fetchers["ma_devenv"] = fetch_ma_devenv_data
        fetchers["ma_train"]  = fetch_ma_train_data

    updated = False
    with ThreadPoolExecutor(max_workers=len(fetchers)) as pool:
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
    """返回所有平台缓存数据，必要时刷新。线程安全。"""
    with _cache_lock:
        age = time.time() - _cache["last_update"]
        roma_keys = ("train", "devenv", "inference")
        if age > CACHE_EXPIRE or not any(_cache[k]["user_data"] for k in roma_keys):
            print(f"缓存过期（{age:.0f}s），重新拉取...")
            refresh_cache()
        else:
            print(f"使用缓存（{age:.0f}s / {CACHE_EXPIRE}s）")
        return {k: dict(_cache[k]) for k in ("train", "devenv", "inference",
                                              "ma_devenv", "ma_train")
                } | {"last_update": _cache["last_update"]}


# ── 路由 ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index_multi.html')


@app.route('/debug/users')
def debug_users():
    """调试接口：查看花名册解析结果和用户映射情况。"""
    sample_uid = request.args.get('uid', '')
    result = {
        'total_entries': len(usr_dict),
        'sample_entries': dict(list(usr_dict.items())[:20]),
        'distinct_leaders': list(set(usr_dict.values())),
    }
    if sample_uid:
        uname, leader = resolve_user(sample_uid)
        result['lookup'] = {'user_id': sample_uid, 'user_name': uname, 'leader': leader}
    return jsonify(result)


@app.route('/data')
def get_data():
    d = get_cached_data()
    return jsonify({
        "train":     d["train"],
        "devenv":    d["devenv"],
        "inference": d["inference"],
        "ma_devenv": d["ma_devenv"],
        "ma_train":  d["ma_train"],
        "quotas":    quota_dict,
        "update_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(d["last_update"])),
    })


if __name__ == '__main__':
    import socket
    _main_parser = argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])
    _main_parser.add_argument('--config', required=True, help='鉴权配置 YAML 文件路径')
    _main_parser.add_argument('--roster', required=True, help='花名册 YAML 文件路径')
    _main_parser.add_argument('--port', type=int, default=5063, help='监听端口（默认 5063）')
    _main_args = _main_parser.parse_args()
    PORT = _main_args.port

    # 检测端口占用
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _chk:
        if _chk.connect_ex(('127.0.0.1', PORT)) == 0:
            raise SystemExit(f"[ERROR] 端口 {PORT} 已被占用，请先停止旧进程（lsof -i :{PORT}）")

    # 获取本机局域网 IP
    try:
        _tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _tmp.connect(('8.8.8.8', 80))
        LOCAL_IP = _tmp.getsockname()[0]
        _tmp.close()
    except Exception:
        LOCAL_IP = '127.0.0.1'

    print(f"[服务] 访问地址：http://{LOCAL_IP}:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
