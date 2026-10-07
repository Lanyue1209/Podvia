"""阶段 3 单元测试：SSML 安全校验（不触网、不调用 API）。

官方白名单：speak(rate/pitch/volume)/break/phoneme/sub/say-as。
阿里云不支持 prosody/emphasis；安全解析拒绝 DOCTYPE/ENTITY（XXE/实体膨胀）。
"""
from __future__ import annotations

import pytest

from podcast_tts.ssml import SSMLError, validate_ssml


def test_plain_speak_passes():
    r = validate_ssml("<speak>你好，世界。</speak>")
    assert not (r.has_rate or r.has_pitch or r.has_volume)
    assert r.text == "<speak>你好，世界。</speak>"  # 原文不改写


def test_speak_attrs_valid():
    r = validate_ssml('<speak rate="1.5" pitch="0.8" volume="80">文本</speak>')
    assert (r.has_rate, r.has_pitch, r.has_volume) == (True, True, True)


@pytest.mark.parametrize("attr,val", [
    ("rate", "0.4"), ("rate", "2.1"), ("rate", "fast"),
    ("pitch", "0"), ("pitch", "x"),
    ("volume", "-1"), ("volume", "101"), ("volume", "50.5"),
])
def test_speak_attrs_out_of_range(attr, val):
    with pytest.raises(SSMLError, match=attr):
        validate_ssml(f'<speak {attr}="{val}">文本</speak>')


def test_speak_unknown_attribute_rejected():
    with pytest.raises(SSMLError, match="不支持的属性"):
        validate_ssml('<speak voice="longanlingxin">文本</speak>')
    with pytest.raises(SSMLError, match="不支持的属性"):
        validate_ssml('<speak effect="robot">文本</speak>')


def test_empty_speak_rejected():
    with pytest.raises(SSMLError, match="内容为空"):
        validate_ssml("<speak>   </speak>")


def test_root_must_be_speak():
    with pytest.raises(SSMLError, match="根节点必须是"):
        validate_ssml("<p>普通标签</p>")


def test_two_roots_invalid_xml():
    with pytest.raises(SSMLError, match="合法 XML"):
        validate_ssml("<speak>a</speak><speak>b</speak>")


def test_malformed_xml_rejected():
    with pytest.raises(SSMLError, match="合法 XML"):
        validate_ssml("<speak>未闭合文本")


# —— break ————————————————————————————————————————————————————————————


def test_break_default_and_explicit():
    validate_ssml("<speak>等一下<break/>好了。</speak>")
    validate_ssml('<speak>等一下<break time="500ms"/>好了。</speak>')
    validate_ssml('<speak>长停顿<break time="2s"/></speak>')


@pytest.mark.parametrize("time_val", ["0s", "11s", "49ms", "10001ms", "abc", "500", "ms"])
def test_break_bad_time(time_val):
    with pytest.raises(SSMLError, match="time"):
        validate_ssml(f'<speak><break time="{time_val}"/></speak>')


def test_break_cumulative_cap():
    # 3 个连续 4s = 12s > 10s（中间无文本）
    with pytest.raises(SSMLError, match="累计"):
        validate_ssml('<speak><break time="4s"/><break time="4s"/><break time="4s"/></speak>')


def test_break_cumulative_resets_with_text():
    # 文本打断后重新累计：4s + 文本 + 4s 合法
    validate_ssml('<speak><break time="4s"/>嗯<break time="4s"/></speak>')


def test_break_must_be_empty():
    with pytest.raises(SSMLError, match="空标签"):
        validate_ssml("<speak><break>x</break></speak>")


# —— phoneme / sub / say-as ———————————————————————————————————————————


def test_phoneme_valid():
    validate_ssml('<speak>去<phoneme alphabet="py" ph="dian3 dang4 hang2">典当行</phoneme></speak>')
    validate_ssml('<speak><phoneme alphabet="cmu" ph="S AY N">sin</phoneme></speak>')


def test_phoneme_bad_alphabet_or_missing_ph():
    with pytest.raises(SSMLError, match="alphabet"):
        validate_ssml('<speak><phoneme alphabet="ipa" ph="a">x</phoneme></speak>')
    with pytest.raises(SSMLError, match="ph"):
        validate_ssml('<speak><phoneme alphabet="py" ph="  ">x</phoneme></speak>')
    with pytest.raises(SSMLError, match="必填"):
        validate_ssml('<speak><phoneme alphabet="py">x</phoneme></speak>')


def test_sub_valid():
    r = validate_ssml('<speak><sub alias="世界卫生组织">WHO</sub></speak>')
    assert "WHO" in r.text


def test_sub_alias_required_and_text_only():
    with pytest.raises(SSMLError, match="alias"):
        validate_ssml("<speak><sub>WHO</sub></speak>")
    with pytest.raises(SSMLError, match="文本"):
        validate_ssml('<speak><sub alias="x"><break/></sub></speak>')


def test_say_as_valid_enum():
    validate_ssml('<speak>电话<say-as interpret-as="telephone">13800000000</say-as></speak>')


def test_say_as_unknown_type_rejected():
    with pytest.raises(SSMLError, match="interpret-as"):
        validate_ssml('<speak><say-as interpret-as="ssn">123</say-as></speak>')


# —— prosody / emphasis：阿里云不支持 ———————————————————————————————————


def test_prosody_explicitly_rejected():
    with pytest.raises(SSMLError, match="prosody"):
        validate_ssml('<speak><prosody rate="slow">文本</prosody></speak>')


def test_emphasis_explicitly_rejected():
    with pytest.raises(SSMLError, match="emphasis"):
        validate_ssml("<speak><emphasis>重点</emphasis></speak>")


def test_nested_tag_in_text_only_rejected():
    with pytest.raises(SSMLError, match="嵌套"):
        validate_ssml(
            '<speak><say-as interpret-as="digits"><break/>123</say-as></speak>'
        )


# —— 安全：XXE / 实体注入 ——————————————————————————————————————————————


def test_doctype_rejected():
    with pytest.raises(SSMLError, match="DOCTYPE"):
        validate_ssml('<?xml version="1.0"?><!DOCTYPE speak [<!ENTITY x "a">]><speak>&x;</speak>')


def test_external_entity_rejected():
    payload = (
        '<!DOCTYPE speak [<!ENTITY xxe SYSTEM "file:///c:/windows/win.ini">]>'
        '<speak>&xxe;</speak>'
    )
    with pytest.raises(SSMLError, match="DOCTYPE|ENTITY"):
        validate_ssml(payload)


def test_billion_laughs_rejected():
    payload = (
        '<!DOCTYPE speak [<!ENTITY a "aa"><!ENTITY b "&a;&a;&a;">]>'
        '<speak>&b;</speak>'
    )
    with pytest.raises(SSMLError, match="DOCTYPE|ENTITY"):
        validate_ssml(payload)


def test_empty_input_rejected():
    with pytest.raises(SSMLError):
        validate_ssml("   ")
