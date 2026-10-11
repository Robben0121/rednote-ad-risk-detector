# -*- coding: utf-8 -*-
"""
小红书美食笔记 · 隐性广告风险判别工具（MVP Demo）

安装依赖：pip install -r requirements.txt
启动：    streamlit run app.py

说明：
- 本 MVP 仅做「单条文本」判别，不涉及爬虫 / 图片 / 账号行为 / 登录 / 数据库 / 批量检测。
- API 配置（api_base / api_key / model_name）在页面侧边栏填写，不硬编码在代码里。
- 使用 OpenAI 兼容接口调用大模型，强制要求返回 JSON。
"""

import json
import re

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# System Prompt（直接写在代码内部，内容来自 system_prompt.txt）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是小红书美食笔记广告风险判别专家。
请严格依据下面的风控特征规则对用户输入的美食笔记做风险判别。

【风控特征规则】
T01 全文零负面评价，没有吐槽、缺点、踩雷、不满意表述 → 偏向重度广告>轻度植入；注意高质量真人笔记也可能全好评，不可单凭该特征判定
T02 完整输出商家营销信息：套餐价格、优惠活动、营业时间、预约提醒 → 偏向重度广告；真人也会记录营业时间分享朋友，不可单条定罪
T03 大量感叹号、营销emoji；高频词：排队王、天花板、快冲、必冲、绝了、宝藏 → 偏向广告；普通人激动也会使用，必须组合其他特征
T04 不同博主同店铺句式结构高度雷同，模板痕迹 → 偏向重度广告；网红店描述会趋同，需要交叉参考
T05 只罗列菜品卖点，缺少“我觉得、不太喜欢、有点咸”个人主观感受 → 偏向重度广告；爱好者也会罗列菜品
T06 存在生活化瑕疵细节：太咸、空调热、某道菜一般、排队久 → 偏向真实分享；置换笔记也会写微小缺点
T07 存在个人随机细节：搭子对话、个人口味、就餐小插曲 → 偏向真实分享；广告脚本一般不会写私人细节
T08 模板化行文结构：开篇引流→菜品介绍→环境→定位标签 → 偏向轻度/重度广告；普通人也会这样写

【输出分级】
真实分享：0‑30分，有主观感受，允许吐槽，商业嫌疑极低
轻度植入：31‑70分，模板化表达，可能置换/霸王餐，可存在少量微小负面
重度广告：71‑100分，付费通稿/MCN脚本，几乎无负面，大量营销信息，缺少个人真实体验

【强制约束】
1. 禁止单条特征直接下定论，综合多条特征给出0‑100整数风险分数。
2. 判别证据必须来自原文，**严禁编造原文不存在的理由**。
3. 边界模糊样本给中间分数，不要强行打0或100。
4. 返回JSON格式，不要输出额外解释。
JSON字段：
{
  "risk_score": 整数0‑100,
  "risk_label": "真实分享"|"轻度植入"|"重度广告",
  "evidence_list": ["证据1（写明对应T编号）","证据2（写明对应T编号）"]
}
只输出JSON，不要markdown、不要多余文字。"""

# ---------------------------------------------------------------------------
# 重度广告模拟样例（用于【填充样例】按钮）
# ---------------------------------------------------------------------------
SAMPLE_NOTE = """姐妹们谁懂啊！这家店也太绝了吧！！排队王天花板真不是吹的🔥🔥🔥
就在市中心商圈，地铁直达超方便！现在限时优惠，双人套餐只要168元，还送两杯饮品，真的太划算了！！
招牌菜必点：招牌烤鱼（外焦里嫩）、蒜蓉粉丝虾、脆皮五花肉，每一道都是颜值担当！
环境ins风，随手一拍都是大片📸 记得提前在大众点评预约哦，周末一定要早点来，不然要等位～
快冲！真的不踩雷！！"""

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """从模型输出中鲁棒地提取 JSON 对象（容忍 markdown 代码块、前后废话）。"""
    if not text:
        raise ValueError("模型返回内容为空")
    s = text.strip()

    # 去掉 markdown 代码块围栏（```json ... ```）
    m = re.match(r"^```[a-zA-Z0-9]*\s*\n?(.*?)\n?```$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()

    # 直接解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 兜底：截取第一个 { 到最后一个 }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("无法从模型输出中解析出有效 JSON")


def call_llm(api_base: str, api_key: str, model_name: str, note_text: str) -> str:
    """调用 OpenAI 兼容接口，返回模型原始文本内容。"""
    client = OpenAI(base_url=api_base.rstrip("/"), api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": note_text},
    ]

    # 优先尝试 response_format 强制 JSON；个别兼容服务不支持时降级重试
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

    content = resp.choices[0].message.content
    if not content:
        raise ValueError("模型返回内容为空")
    return content


# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(page_title="小红书美食笔记 · 隐性广告风险判别", page_icon="🍜", layout="wide")

st.title("🍜 小红书美食笔记 · 隐性广告风险判别")
st.caption("品牌投放运营 · AI 风控判别 MVP —— 仅支持单条文本判别")


# ---------------------------------------------------------------------------
# 侧边栏：API 配置（不硬编码，页面上填写）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔧 API 配置（OpenAI 兼容接口）")
    api_base = st.text_input(
        "api_base",
        value="",
        placeholder="https://api.openai.com/v1",
        help="OpenAI 兼容接口地址，例如 https://api.openai.com/v1",
    )
    api_key = st.text_input(
        "api_key",
        value="",
        type="password",
        help="你的 API Key，仅保存在当前会话，不会落库",
    )
    model_name = st.text_input(
        "model_name",
        value="",
        placeholder="如 gpt-4o / deepseek-chat",
        help="要调用的模型名称",
    )
    st.divider()
    st.caption("提示：仅做单条文本判别；不涉及图片、账号行为、登录、数据库、批量检测。")


# ---------------------------------------------------------------------------
# 输入区
# ---------------------------------------------------------------------------
st.subheader("1️⃣ 输入笔记正文")
note_text = st.text_area(
    "粘贴小红书美食笔记正文",
    height=200,
    key="note_text",
    placeholder="在此粘贴小红书美食笔记正文……",
)


def fill_sample():
    """填充重度广告模拟样例。"""
    st.session_state.note_text = SAMPLE_NOTE


col_fill, col_detect = st.columns([1, 1])
with col_fill:
    st.button("📝 填充样例", on_click=fill_sample, use_container_width=True)
with col_detect:
    detect_clicked = st.button("🔍 开始检测", type="primary", use_container_width=True)

st.caption("提示：本版本仅支持文本输入，图片、账号行为仅作为完整产品方案，当前不参与计算。")


# ---------------------------------------------------------------------------
# 检测逻辑
# ---------------------------------------------------------------------------
if detect_clicked:
    text = note_text.strip()

    # 异常处理 1：输入为空
    if not text:
        st.error("请粘贴笔记文本内容")
    # 异常处理 2：文本过短
    elif len(text) < 30:
        st.warning(f"文本过短（当前 {len(text)} 字，少于 30 字），不足以完成判别，请补充完整笔记内容")
    # 异常处理 3：API 配置缺失
    elif not api_key.strip() or not api_base.strip() or not model_name.strip():
        st.error("请先在左侧「API 配置」中填写 api_base、api_key、model_name")
    else:
        with st.spinner("AI 判别中，请稍候……"):
            try:
                raw = call_llm(api_base, api_key, model_name, text)
                result = extract_json(raw)
                st.session_state["result"] = result
            except Exception as e:
                # 异常处理 4/5：API 调用失败 或 JSON 解析失败
                st.session_state.pop("result", None)
                st.error("AI 服务调用失败或返回无法解析，请稍后重试")
                st.exception(e)


# ---------------------------------------------------------------------------
# 结果展示区
# ---------------------------------------------------------------------------
st.divider()
st.subheader("2️⃣ 判别结果")

result = st.session_state.get("result")
if result:
    # 解析字段（容忍缺失）
    try:
        score = int(result.get("risk_score", 0))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))  # 收敛到 0-100

    label = str(result.get("risk_label", "未知")).strip()
    evidence_list = result.get("evidence_list", [])
    if not isinstance(evidence_list, list):
        evidence_list = []

    # ---- 风险总览卡片 ----
    c_score, c_label, c_level = st.columns(3)
    with c_score:
        st.metric("风险分数", f"{score} / 100")
    st.progress(score / 100.0)

    label_color = {
        "真实分享": "#16a34a",
        "轻度植入": "#f59e0b",
        "重度广告": "#dc2626",
    }.get(label, "#6b7280")
    with c_label:
        st.markdown(
            f"### 风险标签：<span style='color:{label_color};font-weight:700'>{label}</span>",
            unsafe_allow_html=True,
        )

    # 边界样本提示人工复核
    if 35 <= score <= 65:
        st.warning("⚠️ 该样本处于边界模糊区间（35-65），建议人工复核。")

    # ---- 判别证据溯源列表 ----
    st.markdown("#### 📋 判别证据")
    if evidence_list:
        for ev in evidence_list:
            if isinstance(ev, dict):
                # 兼容对象形式
                feat = ev.get("feature") or ev.get("特征编号") or ""
                desc = ev.get("evidence") or ev.get("描述") or ev.get("text") or ""
                feat_tag = f"（{feat}）" if feat else ""
                st.markdown(f"- ✅ {desc}{feat_tag}")
            else:
                st.markdown(f"- ✅ {ev}")
    else:
        st.info("模型未返回判别证据。")

    # ---- 补充提示 ----
    st.info("本 MVP 仅使用文本维度做判别；完整产品还会加入图片、账号行为特征进一步提升准确度")

    # ---- 用户反馈（仅前端打印，不落库） ----
    st.markdown("#### 这条判断准确吗？")
    fb_right, fb_wrong = st.columns(2)
    with fb_right:
        if st.button("👍 判断正确", use_container_width=True):
            st.success("已记录反馈：判断正确（前端演示，未落库）")
    with fb_wrong:
        if st.button("👎 判断错误", use_container_width=True):
            st.warning("已记录反馈：判断错误（前端演示，未落库）")
            st.text_input(
                "可选：简单描述一下哪里判断错了（用于迭代策略）",
                key="feedback_desc",
            )
            if st.session_state.get("feedback_desc"):
                st.caption("感谢反馈，已在前端记录（未落库）。")
