"""Chaihuo Reachy — 柴火基地车 Reachy Mini 智能语音助手."""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from chaihuo_reachy.camera import find_reachy_camera
from chaihuo_reachy.config import Config, load_config
from chaihuo_reachy.dashboard import (
    ChatMessageStore,
    DashboardHub,
    run_websocket_session,
)
from chaihuo_reachy.engine import ConversationEngine
from chaihuo_reachy import daemon_runtime
from chaihuo_reachy.backends.interfaces import (
    playback_gain_from_percent,
    playback_percent_from_gain,
)

if TYPE_CHECKING:
    from chaihuo_reachy.backends.interfaces import AudioBackend, CameraBackend
    from chaihuo_reachy.motion import MotionController
    from reachy_mini import ReachyMini

logger = logging.getLogger("chaihuo_reachy")

_DAEMON_CONNECT_TIMEOUT_S = 2.0
_DAEMON_STARTUP_TIMEOUT_S = 30.0
_DAEMON_POLL_INTERVAL_S = 1.0


class _DaemonStartupError(RuntimeError):
    """The daemon HTTP server is up, but its robot backend failed."""


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>皮皮虾 — 柴火基地车</title>
<style>
html,body{margin:0;padding:0;height:100%;overflow:hidden}
:root{--bg:#0a0e14;--s:#12171f;--b:#1e293b;--t:#c9d1d9;--m:#64748b;--grn:#059669;--glow:#34d399;--warn:#f59e0b;--red:#ef4444}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:var(--bg);color:var(--t);display:flex}
.side{width:260px;background:var(--s);border-right:1px solid var(--b);padding:1.2rem;display:flex;flex-direction:column;gap:1rem;flex-shrink:0;overflow-y:auto}
.side h1{font-size:1.2rem;color:var(--glow)}
.side .sub{color:var(--m);font-size:.7rem;margin-top:-.6rem}
.st-badge{background:var(--b);border-radius:8px;padding:.8rem;text-align:center}
.st-dot{width:14px;height:14px;border-radius:50%;display:inline-block;margin-bottom:.3rem}
.st-dot.idle{background:var(--m)}
.st-dot.listening{background:var(--glow);animation:pulse 1.5s infinite}
.st-dot.thinking{background:var(--warn);animation:pulse .6s infinite}
.st-dot.speaking{background:var(--grn);animation:pulse .8s infinite}
.st-dot.wake_listening{background:#6366f1;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.st-label{font-size:.85rem;font-weight:600;margin-top:.2rem}
.vol label{display:block;font-size:.7rem;color:var(--m);margin-bottom:.2rem}
.vol input{width:100%;accent-color:var(--grn)}
.vol .val{text-align:right;font-size:.7rem;color:var(--glow)}
.btn{width:100%;padding:.4rem .8rem;border:1px solid var(--b);border-radius:6px;background:var(--s);color:var(--t);cursor:pointer;font-size:.75rem;transition:.2s}
.btn:hover{border-color:var(--grn)}
.btn.on{background:var(--grn);border-color:var(--grn);color:#fff}
.btn.sm{padding:.2rem .5rem;font-size:.65rem;margin-bottom:.15rem}
.side .foot{margin-top:auto;font-size:.6rem;color:var(--m);border-top:1px solid var(--b);padding-top:.6rem}
.ws-ok{color:var(--glow)}.ws-err{color:var(--red)}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.cam{display:flex;gap:.8rem;padding:.5rem 1rem;background:var(--s);border-bottom:1px solid var(--b);align-items:center;flex-shrink:0}
.cam-box{width:180px;height:100px;border-radius:6px;overflow:hidden;background:#000;border:1px solid var(--b);flex-shrink:0;position:relative}
.cam-box img{width:100%;height:100%;object-fit:cover}
.cam-box .live{position:absolute;top:4px;right:6px;font-size:.55rem;color:#0f0;background:rgba(0,0,0,.5);padding:1px 4px;border-radius:3px}
.cam-info{font-size:.7rem;color:var(--m)}
.asr{min-height:2rem;padding:.5rem 1rem;font-size:.95rem;color:var(--glow);background:rgba(5,150,105,.05);border-bottom:1px solid var(--b);font-style:italic;transition:.2s;flex-shrink:0}
.asr.active{color:#fff;background:rgba(5,150,105,.1)}
.chat{flex:1 1 0;overflow-y:auto;padding:1rem;display:block;min-height:0}
.msg{max-width:75%;padding:.5rem .8rem;border-radius:8px;line-height:1.5;font-size:.85rem;animation:fade .3s;margin-bottom:.6rem}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
.msg.u{margin-left:auto;background:var(--grn);color:#fff}
.msg.a{margin-right:auto;background:var(--s);border:1px solid var(--b)}
.msg.a.stream{border-color:var(--grn);box-shadow:0 0 6px rgba(5,150,105,.12)}
.msg .who{font-size:.6rem;opacity:.5;margin-bottom:.15rem}
.msg .ts{font-size:.55rem;opacity:.3;margin-top:.15rem}
.bar{padding:.4rem 1rem;background:var(--s);border-top:1px solid var(--b);font-size:.65rem;color:var(--m);display:flex;gap:1rem;flex-shrink:0}
.debug-bar{display:flex;gap:.4rem;padding:.5rem 1rem;background:var(--s);border-top:1px solid var(--b);align-items:flex-end;flex-shrink:0}
.debug-bar textarea{flex:1;background:var(--bg);border:1px solid var(--b);border-radius:6px;color:var(--t);padding:.4rem .6rem;font-size:.8rem;resize:none;outline:none;font-family:inherit;min-height:2rem;max-height:6rem;transition:border-color .2s}
.debug-bar textarea:focus{border-color:var(--grn)}
.debug-bar .db-cam{display:flex;align-items:center;gap:.2rem;font-size:.7rem;color:var(--m);cursor:pointer;white-space:nowrap;user-select:none}
.debug-bar .db-cam input{accent-color:var(--grn)}
.debug-bar .db-send{padding:.4rem 1rem;background:var(--grn);border:none;border-radius:6px;color:#fff;cursor:pointer;font-size:.75rem;font-weight:600;white-space:nowrap;transition:opacity .2s}
.debug-bar .db-send:hover{opacity:.85}
.debug-bar .db-send:disabled{opacity:.4;cursor:not-allowed}
.asr-diag{background:var(--b);border-radius:8px;padding:.6rem .8rem;font-size:.7rem}
.asr-diag .row{display:flex;justify-content:space-between;align-items:center;margin-bottom:.25rem}
.asr-diag .row:last-child{margin-bottom:0}
.asr-diag .lbl{color:var(--m)}
.asr-diag .val{color:var(--t);font-weight:600;font-family:monospace}
.asr-diag .val.ok{color:var(--glow)}.asr-diag .val.warn{color:var(--warn)}.asr-diag .val.err{color:var(--red)}
.level-bar{width:100%;height:8px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden;margin:.2rem 0}
.level-bar .fill{height:100%;border-radius:4px;transition:width .15s,background .3s}
.level-bar .fill.low{background:var(--m);width:10%}
.level-bar .fill.mid{background:var(--grn);width:40%}
.level-bar .fill.high{background:var(--glow);width:70%}
.level-bar .fill.clip{background:var(--red);width:95%}
.asr-hist{max-height:120px;overflow-y:auto;font-size:.65rem}
.asr-hist .entry{padding:.15rem .3rem;margin:.1rem 0;border-radius:4px;background:rgba(255,255,255,.02);display:flex;gap:.3rem;align-items:center}
.asr-hist .entry .t{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
</style>
</head>
<body>
<div class="side">
  <h1>🚐 皮皮虾</h1>
  <p class="sub">柴火基地车 Reachy Mini</p>
  <div id="connBanner" style="background:#ef444420;border:2px solid #ef4444;border-radius:8px;padding:.5rem;text-align:center;margin-bottom:.5rem;font-size:.75rem;color:#ef4444;display:block">
    🔴 未连接 — 请刷新页面 (Cmd+Shift+R)
  </div>
<div class="st-badge">
    <div class="st-dot idle" id="dot"></div>
    <div class="st-label" id="stLabel">连接中…</div>
  </div>
  <div class="asr-diag" id="asrDiag">
    <div class="row"><span class="lbl">🎤 输入电平</span><span class="val" id="lvDb">-- dB</span></div>
    <div class="level-bar"><div class="fill low" id="lvFill"></div></div>
    <div class="row"><span class="lbl">🗣 VAD</span><span class="val" id="vadSt">--</span></div>
    <div class="row"><span class="lbl">🔗 ASR WS</span><span class="val" id="asrWs">--</span></div>
    <div class="row"><span class="lbl">📝 本轮</span><span class="val" id="curPart" style="font-size:.65rem">--</span></div>
    <div class="row"><span class="lbl">✅ 最终</span><span class="val ok" id="lastFinal" style="font-size:.65rem">--</span></div>
  </div>
  <div class="asr-diag" style="margin-top:.2rem">
    <div class="row" style="margin-bottom:.3rem"><span class="lbl">📜 识别记录</span></div>
    <div class="asr-hist" id="asrHist"><div style="color:var(--m);text-align:center">等待识别...</div></div>
  </div>
  <div class="asr-diag" style="margin-top:.2rem" id="locCard">
    <div class="row" style="margin-bottom:.3rem"><span class="lbl">📍 当前位置</span><span class="val" id="locSrc" style="font-size:.6rem">--</span></div>
    <div style="font-size:.65rem;color:var(--t);line-height:1.5" id="locDetail">
      <div style="color:var(--m)">定位中...</div>
    </div>
    <div class="row" style="margin-top:.3rem">
      <input type="text" id="locManualLat" placeholder="纬度" style="width:50%;background:var(--bg);border:1px solid var(--b);border-radius:4px;color:var(--t);padding:2px 4px;font-size:.6rem">
      <input type="text" id="locManualLon" placeholder="经度" style="width:50%;background:var(--bg);border:1px solid var(--b);border-radius:4px;color:var(--t);padding:2px 4px;font-size:.6rem;margin-left:.2rem">
    </div>
    <button class="btn sm" onclick="setManualLoc()" style="margin-top:.2rem">📍 手动设置位置</button>
  </div>
  <div class="vol">
    <label>🔊 扬声器音量</label>
    <input type="range" id="volS" min="0" max="100" value="70" disabled>
    <div class="val" id="volV">--</div>
  </div>
  <button class="btn on" id="btnW" onclick="toggleWake()">🎤 唤醒词: 开启</button>
  <button class="btn" id="btnSearch" onclick="toggleSearch()">🌐 联网搜索: 关闭</button>
  <button class="btn" onclick="snapshot()">📷 拍照分析</button>
  <div style="margin-top:.3rem;padding-top:.3rem;border-top:1px solid rgba(255,255,255,.1)">
    <div style="font-size:.6rem;color:var(--m);margin-bottom:.2rem">🤖 机器人动作</div>
    <button class="btn sm" onclick="sendMotion('motion_dance',{style:'happy'})">💃 跳舞</button>
    <button class="btn sm" onclick="sendMotion('motion_nod')">🙆 点头</button>
    <button class="btn sm" onclick="sendMotion('motion_shake_head')">🙅 摇头</button>
    <button class="btn sm" onclick="sendMotion('motion_wave')">🐜 挥天线</button>
    <button class="btn sm" onclick="sendMotion('motion_pose',{action:'sleep'})">😴 休眠</button>
    <button class="btn sm" onclick="sendMotion('motion_pose',{action:'wake_up'})">🤖 站起</button>
  </div>
  <div class="foot">
    <div id="audioInfo">🎤 音频设备解析中…</div>
    <div>📷 Reachy Mini Camera</div>
    <div id="sdkInfo">🔌 Standalone 模式</div>
    <div>🤖 qwen-turbo · qwen3-tts</div>
    <div style="margin-top:.3rem"><span id="wsDot">●</span> <span id="wsTxt">连接中</span> · <span id="evtN">0</span>事件</div>
    <div style="font-size:.55rem;color:var(--glow);margin-top:.2rem" id="stDbg"></div>
    <div style="font-size:.6rem;color:var(--m);margin-top:.2rem" id="motionState">🤖 就绪</div>
  </div>
</div>
<div class="main">
  <div class="cam">
    <div class="cam-box">
      <img id="camImg" src="/camera/stream" onload="this.style.display='block';document.getElementById('liveTag').style.display='block'">
      <div class="live" id="liveTag" style="display:none">LIVE</div>
    </div>
    <div class="cam-box" id="rearViewBox" style="display:none">
      <img id="rearImg" src="" onload="document.getElementById('rearViewBox').style.display='block'">
      <div class="live" style="position:absolute;top:4px;right:6px;font-size:.55rem;color:#f90;background:rgba(0,0,0,.5);padding:1px 4px;border-radius:3px">后视</div>
    </div>
    <div class="cam-info">Reachy Mini Camera — 实时MJPEG</div>
  </div>
  <div style="display:flex;align-items:center;gap:.5rem;padding:.3rem 1rem;background:var(--s);border-bottom:1px solid var(--b);flex-shrink:0">
    <div id="emotionFace" style="font-size:1.5rem;transition:.3s">😐</div>
    <div class="asr" id="asrBox" style="min-height:2.5rem;font-size:1.1rem;flex:1;border:none;background:transparent"></div>
  </div>
  <div class="chat" id="chat"></div>
  <div class="debug-bar">
    <textarea id="dbInput" placeholder="💬 输入文字直接对话 (Enter 发送，Shift+Enter 换行)..." rows="1" onkeydown="onDbKey(event)"></textarea>
    <label class="db-cam" title="附带当前摄像头画面"><input type="checkbox" id="dbCam" checked>📷</label>
    <button class="db-send" id="dbSend" onclick="sendDebug()">发送</button>
  </div>
  <div class="bar"><span id="uptime">00:00</span> · <span id="msgN">0 条对话</span> · <span style="color:var(--glow)">⌨ 下方输入框可直接对话</span></div>
</div>
<script>
var ws, msgN=0, t0=Date.now(), evtN=0, curStreamId=null,finalTimer=null,turnActive=false;
var seenSeq=new Set(),seenOrder=[];
var labels={idle:'💤',listening:'🎤',thinking:'🤔',speaking:'🔊',wake_listening:'👂'};
var lbl={idle:'就绪',listening:'聆听中',thinking:'思考中',speaking:'回复中',wake_listening:'等待唤醒'};

function $(id){return document.getElementById(id);}
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}

// Dead-simple: every event updates the UI directly
function onMsg(m){
  if(m.seq!=null){
    if(seenSeq.has(m.seq))return;
    seenSeq.add(m.seq);seenOrder.push(m.seq);
    if(seenOrder.length>256)seenSeq.delete(seenOrder.shift());
  }
  evtN++; $('evtN').textContent=evtN;
  if(m.type==='state'){
    $('dot').className='st-dot '+m.state;
    $('stLabel').textContent=labels[m.state]+' '+lbl[m.state];
    $('stDbg').textContent=m.state;
    if((m.state==='thinking'||m.state==='speaking')&&!curStreamId&&turnActive){
      curStreamId='s'+Date.now();
      addMsg('a', m.state==='thinking'?'🤔 思考中...':'', true, curStreamId);
    }
    // When transitioning from thinking → speaking, update the placeholder
    if(m.state==='speaking'&&curStreamId){
      var el=document.getElementById(curStreamId);
      if(el){var c=el.querySelector('.c');if(c&&c.textContent==='🤔 思考中...')c.textContent='';}
    }
    if(m.state==='idle'){
      turnActive=false;
      if(curStreamId){
        var el=document.getElementById(curStreamId);
        if(el)el.classList.remove('stream');
        curStreamId=null;
      }
    }
  }else if(m.type==='asr_status'){
    $('asrBox').textContent='🎤 '+m.status; $('asrBox').classList.add('active');
  }else if(m.type==='transcript'){
    if(m.final){
      if(m.text){
        turnActive=true;  // real user speech — allow AI bubbles
        $('asrBox').textContent='✅ '+m.text;$('asrBox').classList.add('active');
        addMsg('u',m.text,false);
        // Update diagnostics
        $('curPart').textContent='✅ '+m.text;$('curPart').className='val ok';
        $('lastFinal').textContent=m.text;$('lastFinal').className='val ok';
        // Add to ASR history
        var hist=$('asrHist'),entryDiv=document.createElement('div');
        entryDiv.className='entry';
        entryDiv.innerHTML='<span style="color:var(--glow);font-size:.6rem">#'+(m.seq||'?')+'</span><span class="t" title="'+esc(m.text)+'">'+esc(m.text.slice(0,30))+'</span>';
        var firstKid=hist.firstElementChild;
        if(firstKid&&firstKid.style&&firstKid.style.color==='rgb(100, 116, 139)')hist.innerHTML='';
        hist.insertBefore(entryDiv,hist.firstChild);
        while(hist.children.length>20)hist.removeChild(hist.lastChild);
        if(finalTimer)clearTimeout(finalTimer);
        finalTimer=setTimeout(function(){
          var seq=m.seq;
          if(seq===m.seq&&$('asrBox').textContent==='✅ '+m.text){
            $('asrBox').classList.remove('active');
          }
        },5000);
      }
    }else if(m.text){
      $('asrBox').textContent='🎤 '+m.text; $('asrBox').classList.add('active');
      $('curPart').textContent='📝 '+m.text;$('curPart').className='val';
    }
  }else if(m.type==='llm_token'){
    if(curStreamId){
      var el=document.getElementById(curStreamId);
      if(el){el.querySelector('.c').textContent+=m.text;var chat=$('chat');var atBottom=chat.scrollHeight-chat.scrollTop-chat.clientHeight<60;if(atBottom)chat.scrollTop=chat.scrollHeight;}
    }else{
      // Fallback: create streaming msg if state 'speaking' was missed
      curStreamId='s'+Date.now();
      addMsg('a',m.text,true,curStreamId);
    }
  }else if(m.type==='emotion'){
    // LLM emotion → face emoji
    var emojiMap={happy:'😊',sad:'😢',curious:'🤔',excited:'🤩',neutral:'😐',surprised:'😮',angry:'😠',love:'😍',sleepy:'😴',laughing:'😆',winking:'😉',thinking:'🤔'};
    var face=emojiMap[m.emotion]||'😊';
    $('emotionFace').textContent=face;
  }else if(m.type==='volume'){
    $('volS').value=m.volume;$('volV').textContent=m.volume+'%';
  }else if(m.type==='wake_word'){
    $('btnW').textContent='🎤 唤醒词: '+(m.enabled?'开启':'关闭');
    $('btnW').className='btn'+(m.enabled?' on':'');
  }else if(m.type==='search'){
    $('btnSearch').textContent='🌐 联网搜索: '+(m.enabled?'开启':'关闭');
    $('btnSearch').className='btn'+(m.enabled?' on':'');
  }else if(m.type==='snapshot_result'){
    if(m.jpeg){
      if(m.label==='rear'){
        // Show rear-view camera snapshot
        $('rearImg').src='data:image/jpeg;base64,'+m.jpeg;
        $('rearViewBox').style.display='block';
        $('camImg').src='/camera/stream';  // Restore front camera stream
        $('liveTag').style.display='block';
        // Auto-hide rear view after 15s
        setTimeout(function(){$('rearViewBox').style.display='none';},15000);
      }else{
        $('camImg').src='data:image/jpeg;base64,'+m.jpeg;$('liveTag').style.display='none';
      }
    }
  }else if(m.type==='runtime_status'){
    var a=m.audio;
    if(a&&a.input&&a.output){
      var chInfo='';
      if(a.input.stream_channels)chInfo=' (ch'+a.input.stream_channels+')';
      $('audioInfo').textContent='🎤 ['+a.input.index+'] '+a.input.name+' · '+a.input.max_channels+'in/'+a.output.max_channels+'out'+chInfo;
    }else{$('audioInfo').textContent='🎤 音频未就绪';}
    $('sdkInfo').textContent=m.sdk_connected?'🔌 Reachy SDK 已连接':'🔌 Standalone 模式';
    // Update location display
    if(m.location){
      var loc=m.location;
      var srcLabel={'gpsd':'🛰 GPS卫星','browser':'📱 设备定位','amap_ip':'🌐 高德IP城市定位','session_user':'📍 会话位置','configured_fallback':'📌 默认城市','manual':'📍 手动设置','unavailable':'❌ 无信号'};
      $('locSrc').textContent=srcLabel[loc.source]||loc.source;
      $('locSrc').className='val '+(loc.source==='gpsd'?'ok':loc.source==='unavailable'?'err':'');
      var detail=(loc.lat!=null&&loc.lon!=null)?('坐标: '+loc.lat.toFixed(4)+'°, '+loc.lon.toFixed(4)+'°'):'精度: '+(loc.precision==='city'?'城市级':'未提供坐标');
      if(loc.accuracy_m!=null){detail+=' (精度 ±'+loc.accuracy_m.toFixed(0)+'m)';}
      if(loc.address){detail+='<br>'+esc(loc.address);}
      if(loc.altitude_m!=null){detail+='<br>海拔: '+loc.altitude_m.toFixed(0)+'m';}
      if(loc.speed_kmh!=null&&loc.speed_kmh>0.5){detail+='<br>速度: '+loc.speed_kmh.toFixed(1)+'km/h';}
      var updated=loc.observed_at?new Date(loc.observed_at):(loc.timestamp?new Date(loc.timestamp*1000):null);
      if(updated)detail+='<br><span style="color:var(--m);font-size:.6rem">更新: '+updated.toLocaleTimeString()+'</span>';
      $('locDetail').innerHTML=detail;
    }
    // Update ASR diagnostics
    if(m.audio_level_db!=null){
      var db=m.audio_level_db, dbTxt=db.toFixed(1)+' dB';
      $('lvDb').textContent=dbTxt;
      var fill=$('lvFill'), pct=Math.min(100,Math.max(0,(db+60)/60*100));
      fill.style.width=pct+'%';
      fill.className='fill '+(db<-40?'low':db<-20?'mid':db<-3?'high':'clip');
    }
  }else if(m.type==='audio_level'){
    var db=m.db;if(db==null)return;
    $('lvDb').textContent=db.toFixed(1)+' dB';
    var fill=$('lvFill'), pct=Math.min(100,Math.max(0,(db+60)/60*100));
    fill.style.width=pct+'%';
    fill.className='fill '+(db<-40?'low':db<-20?'mid':db<-3?'high':'clip');
    // Camera frame age warning
    var age=m.frame_age_s;
    var tag=$('liveTag');
    if(age!=null&&age>3.0){
      tag.textContent='⚠ '+(age||0).toFixed(0)+'s';
      tag.style.color='#f59e0b';
      $('camImg').style.opacity='0.4';
    }else if(age!=null&&age>=0){
      tag.textContent='LIVE';
      tag.style.color='#0f0';
      $('camImg').style.opacity='1';
    }
  }else if(m.type==='vad_event'){
    if(m.event==='speech_started'){
      $('vadSt').textContent='🟢 检测到语音';$('vadSt').className='val ok';
    }else if(m.event==='speech_stopped'){
      $('vadSt').textContent='🛑 语音结束'+(m.duration_s?(' ('+m.duration_s.toFixed(1)+'s)'):'');$('vadSt').className='val';
    }
  }else if(m.type==='asr_status'){
    if(m.connected){$('asrWs').textContent='🟢 已连接';$('asrWs').className='val ok';}
    else{$('asrWs').textContent='🔴 断开';$('asrWs').className='val err';}
  }else if(m.type==='asr_history_replay'){
    var items=m.items||[];
    var hist=$('asrHist');hist.innerHTML='';
    if(!items||items.length===0){hist.innerHTML='<div style="color:var(--m);text-align:center">等待识别...</div>';return;}
    for(var i=items.length-1;i>=0;i--){
      var entry=items[i];
      var div=document.createElement('div');div.className='entry';
      div.innerHTML='<span style="color:var(--glow);font-size:.6rem">#'+entry.seq+'</span><span class="t" title="'+esc(entry.text)+'">'+esc(entry.text.slice(0,30))+'</span>';
      hist.appendChild(div);
    }
  }else if(m.type==='motion_status'){
    var s=m.action||'idle';
    var labels={idle:'🤖 就绪',dancing:'💃 跳舞中…',nodding:'🙆 点头中…',shaking:'🙅 摇头中…',waving:'🐜 挥天线…',sleeping:'😴 休眠中…',ready:'✅ 已就绪'};
    var el=document.getElementById('motionState');
    if(el)el.textContent=labels[s]||'🤖 '+s;
  }
}

function addMsg(role,txt,stream,id){
  var d=document.createElement('div'),ts=new Date();
  d.className='msg '+(role==='u'?'u':'a')+(stream?' stream':'');
  if(id)d.id=id;
  d.innerHTML='<div class="who">'+(role==='u'?'🧑 你':'🤖 皮皮虾')+'</div><div class="c">'+esc(txt)+'</div><div class="ts">'+ts.getHours().toString().padStart(2,'0')+':'+ts.getMinutes().toString().padStart(2,'0')+'</div>';
  var chat=$('chat');
  // Only auto-scroll when user is already near the bottom (within 60px).
  // Otherwise they are reading history and we shouldn't yank them down.
  var atBottom=chat.scrollHeight-chat.scrollTop-chat.clientHeight<60;
  chat.appendChild(d);
  if(atBottom)chat.scrollTop=chat.scrollHeight;
  msgN++;$('msgN').textContent=msgN+' 条对话';
  return d;
}

// ── Debug text chat ────────────────────────────────────────────────
function getCamBase64(){
  try{
    var img=$('camImg');
    var c=document.createElement('canvas');
    c.width=img.naturalWidth||640;c.height=img.naturalHeight||360;
    c.getContext('2d').drawImage(img,0,0);
    return c.toDataURL('image/jpeg',0.75).split(',')[1];
  }catch(e){return null;}
}

var dbSending=false;
async function sendDebug(){
  if(dbSending)return;
  var text=$('dbInput').value.trim();
  if(!text)return;
  dbSending=true;
  var btn=$('dbSend');btn.disabled=true;btn.textContent='…';
  // Show user message
  addMsg('u',text,false);
  $('dbInput').value='';$('dbInput').style.height='auto';
  // Build request
  var body={text:text};
  if($('dbCam').checked){
    var b64=getCamBase64();
    if(b64)body.image_base64=b64;
  }
  // Create streaming placeholder
  var sid='db'+Date.now();
  addMsg('a','',true,sid);
  try{
    var resp=await fetch('/debug/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    var data=await resp.json();
    var el=document.getElementById(sid);
    if(el){
      el.classList.remove('stream');
      el.querySelector('.c').textContent=data.reply||'(空)';
      if(data.memory_context)el.querySelector('.c').title='记忆:'+data.memory_context.slice(0,200);
    }
  }catch(e){
    var el=document.getElementById(sid);
    if(el){el.classList.remove('stream');el.querySelector('.c').textContent='❌ 请求失败: '+e.message;}
  }
  dbSending=false;btn.disabled=false;btn.textContent='发送';
  $('dbInput').focus();
}

function onDbKey(e){
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendDebug();}
}

function connect(){
  // Self-test: verify DOM and WS work
  $('stLabel').textContent='⏳ 连接中';
  $('asrBox').textContent='🔗 正在连接WebSocket...';

  var wsProtocol=window.location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(wsProtocol+'//'+window.location.host+'/ws');
  ws.onopen=function(){
    $('wsDot').className='ws-ok';$('wsTxt').textContent='已连接';$('volS').disabled=false;
    var b=$('connBanner');if(b){b.style.background='#05966920';b.style.borderColor='var(--grn)';b.style.color='var(--glow)';b.textContent='🟢 已连接 — 开始对话吧!';}
    ws.send(JSON.stringify({type:'get_volume'}));
    ws.send(JSON.stringify({type:'get_state'}));
    ws.send(JSON.stringify({type:'get_wake_word'}));
    ws.send(JSON.stringify({type:'get_search'}));
  };
  ws.onclose=function(){$('wsDot').className='ws-err';$('wsTxt').textContent='断开';setTimeout(connect,1500);};
  ws.onerror=function(){$('wsDot').className='ws-err';$('wsTxt').textContent='错误';};
  ws.onmessage=function(e){
    try{var m=JSON.parse(e.data);onMsg(m);}catch(err){console.log('WS err',err);}
  };
}

$('volS').addEventListener('input',function(){
  var v=parseInt(this.value);$('volV').textContent=v+'%';
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'set_volume',volume:v}));
});

function toggleWake(){
  var on=$('btnW').className.indexOf('on')===-1;
  $('btnW').textContent='🎤 唤醒词: '+(on?'开启':'关闭');
  $('btnW').className='btn'+(on?' on':'');
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'set_wake_word',enabled:on}));
}

function toggleSearch(){
  var on=$('btnSearch').className.indexOf('on')===-1;
  $('btnSearch').textContent='🌐 联网搜索: '+(on?'开启':'关闭');
  $('btnSearch').className='btn'+(on?' on':'');
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'set_search',enabled:on}));
}

function snapshot(){if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'snapshot'}));}
function sendMotion(type,data){if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:type,...(data||{})}));}
function setManualLoc(){
  var lat=parseFloat($('locManualLat').value);
  var lon=parseFloat($('locManualLon').value);
  if(isNaN(lat)||isNaN(lon)){alert('请输入有效的经纬度（如 22.5431, 113.9544）');return;}
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'set_location',lat:lat,lon:lon}));
}

// ── Browser geolocation (macOS CoreLocation → Wi-Fi positioning) ──
var _geoWatchId=null;
function startGeoTracking(){
  if(!navigator.geolocation){$('locSrc').textContent='⚠️ 不支持';return;}
  // One-shot immediate position
  navigator.geolocation.getCurrentPosition(function(pos){
    sendGeoPos(pos);
  },function(err){
    $('locSrc').textContent='⚠️ 定位被拒绝';
    console.log('Geolocation error:',err.message);
  },{enableHighAccuracy:true,timeout:10000,maximumAge:30000});
  // Continuous tracking
  _geoWatchId=navigator.geolocation.watchPosition(function(pos){
    sendGeoPos(pos);
  },function(err){
    console.log('Geolocation watch error:',err.message);
  },{enableHighAccuracy:true,timeout:15000,maximumAge:10000});
}
function sendGeoPos(pos){
  if(!ws||ws.readyState!==1)return;
  ws.send(JSON.stringify({
    type:'browser_location',
    lat:pos.coords.latitude,
    lon:pos.coords.longitude,
    accuracy:pos.coords.accuracy,
    altitude:pos.coords.altitude,
    heading:pos.coords.heading,
    speed:pos.coords.speed
  }));
}
startGeoTracking();

connect();
setInterval(function(){
  var s=Math.floor((Date.now()-t0)/1000);
  $('uptime').textContent='⏱ '+String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
},5000);
</script>
</body>
</html>"""

DASHBOARD_HTML = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in (
        "httpx",
        "websockets",
        "chromadb",
        "urllib3",
        "sounddevice",
        "dashscope",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── MJPEG camera stream helper ──────────────────────────────────────────────


class _MJPEGStream:
    """Thread-safe MJPEG streamer — supports both SDK camera backend and OpenCV."""

    def __init__(
        self,
        camera_device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
        camera_backend: "CameraBackend | None" = None,
    ):
        self._device = camera_device
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = camera_backend  # SDK backend takes priority
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_jpeg: bytes = b""
        self._frame_time = 0.0

    def start(self) -> bool:
        # SDK backend mode
        if self._backend is not None:
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop_backend, daemon=True, name="mjpeg-capture"
            )
            self._thread.start()
            logger.info("MJPEG stream started: SDK backend")
            return True

        # Direct OpenCV mode
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            logger.error("MJPEG stream: cannot open camera %s", self._device)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        for _ in range(5):
            self._cap.read()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="mjpeg-capture"
        )
        self._thread.start()
        logger.info(
            "MJPEG stream started: %dx%d@%d", self._width, self._height, self._fps
        )
        return True

    def stop(self) -> None:
        self._running = False
        # Don't block on thread join — the capture thread may be stuck in
        # a blocking camera read. The thread is a daemon and will die with
        # the process. Release the camera to unblock any pending read.
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("MJPEG stream stopped")

    def _capture_loop(self) -> None:
        interval = 1.0 / max(self._fps, 1)
        while self._running:
            ret, frame = self._cap.read() if self._cap else (False, None)
            if ret and frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()
                    self._frame_time = time.monotonic()
            time.sleep(interval)

    def _capture_loop_backend(self) -> None:
        """Capture loop for SDK camera backend."""
        interval = 1.0 / max(self._fps, 1)
        _failures = 0
        _last_warn = 0.0
        _quality_checked = 0
        while self._running and self._backend is not None:
            jpeg = self._backend.capture_jpeg(quality=75)
            if jpeg and jpeg[:2] == b"\xff\xd8":
                # Quick periodic darkness check (every ~30 frames) — only for
                # logging, never blocks the frame from reaching the stream.
                _quality_checked += 1
                if _quality_checked % 30 == 0:
                    try:
                        arr = np.frombuffer(jpeg, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        if img is not None and img.size > 0:
                            dark_ratio = float((img < 20).mean())
                            if dark_ratio > 0.90:
                                logger.warning(
                                    "📷 摄像头画面偏暗 (%.0f%% 像素<20) — 检查镜头是否被遮挡",
                                    dark_ratio * 100,
                                )
                    except Exception:
                        pass
                with self._lock:
                    self._latest_jpeg = jpeg
                    self._frame_time = time.monotonic()
                _failures = 0
            else:
                _failures += 1
                now = time.monotonic()
                if _failures >= 5 and now - _last_warn > 5.0:
                    _last_warn = now
                    logger.warning(
                        "📷 MJPEG: %d 连续帧获取失败 — 摄像头可能断开或被遮挡",
                        _failures,
                    )
            time.sleep(interval)

    def get_frame(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    async def capture_fresh(self, timeout_s: float = 1.5) -> bytes | None:
        """Wait for a frame produced after this capture request."""
        requested_at = time.monotonic()
        deadline = requested_at + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame_time > requested_at and self._latest_jpeg:
                    return self._latest_jpeg
            await asyncio.sleep(0.02)
        return None

    @property
    def frame_age_s(self) -> float | None:
        with self._lock:
            captured_at = self._frame_time
        if captured_at <= 0:
            return None
        return max(0.0, time.monotonic() - captured_at)

    @property
    def is_active(self) -> bool:
        return self._running and (time.monotonic() - self._frame_time) < 3.0

    @property
    def is_running(self) -> bool:
        return self._running


async def generate_mjpeg(stream: _MJPEGStream):
    """Async generator for MJPEG HTTP response."""
    boundary = b"--mjpeg-boundary"
    while stream.is_running:
        frame = stream.get_frame()
        if frame:
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
            yield frame + b"\r\n"
        await asyncio.sleep(0.1)


# ── Dashboard runner ─────────────────────────────────────────────────────────


async def run_dashboard(
    cfg: Config,
    *,
    sdk_status: dict[str, Any] | None = None,
    stop_event: threading.Event | None = None,
    audio_backend: "AudioBackend | None" = None,
    camera_backend: "CameraBackend | None" = None,
    reachy: "ReachyMini | None" = None,
    motion: "MotionController | None" = None,
    beat_dance: Any | None = None,
    manage_reachy_lifecycle: bool = False,
) -> None:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import (
        HTMLResponse,
        JSONResponse,
        Response,
        StreamingResponse,
    )
    import uvicorn

    app = FastAPI(title="皮皮虾 Dashboard")
    started_at = time.time()

    # Resolve and open exactly one front-camera backend.  The MJPEG service,
    # VLM snapshots, and the engine all share this instance.
    if camera_backend is None:
        from chaihuo_reachy.backends.factory import create_camera_backend

        if cfg.camera_device == "auto":
            cfg.camera_device = find_reachy_camera(cfg.camera_device)
        camera_backend = create_camera_backend(cfg)

    engine = ConversationEngine(
        cfg,
        audio_backend=audio_backend,
        camera_backend=camera_backend,
        motion=motion,
        beat_dance=beat_dance,
    )

    mjpeg = _MJPEGStream(camera_backend=camera_backend, fps=10)
    mjpeg_ok = mjpeg.start()
    engine.set_camera_snapshot_provider(mjpeg.capture_fresh)

    # ── Broadcast infrastructure ──
    hub = DashboardHub()
    chat = ChatMessageStore(
        message_limit=cfg.chat_history_limit,
        capture_limit=cfg.capture_history_limit,
    )
    sdk_status = dict(sdk_status or {"sdk_connected": False, "mode": "standalone"})

    def broadcast(msg: dict[str, Any]) -> None:
        hub.publish(msg)

    engine.on_state_change(lambda s: broadcast({"type": "state", "state": s}))
    engine.on_transcript(
        lambda t, f: broadcast({"type": "transcript", "text": t, "final": f})
    )
    engine.on_asr_status(
        lambda status: broadcast({"type": "asr_status", "status": status})
    )
    engine.on_emotion(lambda e: broadcast({"type": "emotion", "emotion": e}))

    def on_turn_event(event: dict[str, Any]) -> None:
        event_type = event["type"]
        turn_id = str(event.get("turn_id") or "")
        if event_type == "turn_begin":
            user, assistant = chat.begin_turn(
                turn_id,
                str(event.get("text") or ""),
                source=str(event.get("source") or "unknown"),
                client_message_id=str(event.get("client_message_id") or ""),
            )
            broadcast({"type": "chat_message_upsert", "message": user})
            broadcast({"type": "chat_message_upsert", "message": assistant})
        elif event_type == "chat_message_delta":
            message = chat.append_delta(turn_id, str(event.get("delta") or ""))
            if message:
                broadcast(
                    {
                        "type": "chat_message_delta",
                        "message_id": message["id"],
                        "turn_id": turn_id,
                        "delta": str(event.get("delta") or ""),
                    }
                )
        elif event_type == "turn_status":
            status_value = str(event.get("status") or "")
            message = chat.set_status(turn_id, status_value)
            broadcast(
                {"type": "turn_status", "turn_id": turn_id, "status": status_value}
            )
            if message:
                broadcast({"type": "chat_message_upsert", "message": message})
        elif event_type == "turn_final":
            message = chat.finalize(
                turn_id,
                str(event.get("text") or ""),
                sources=list(event.get("sources") or []),
                error=str(event.get("error")) if event.get("error") else None,
            )
            if message:
                broadcast({"type": "chat_message_upsert", "message": message})

    def on_snapshot(jpeg: bytes, label: str) -> None:
        display_label = "车外后视" if label == "rear" else "Reachy 前置"
        capture_id, message = chat.attach_capture(
            engine.current_turn_id,
            jpeg,
            label=display_label,
        )
        if message:
            broadcast({"type": "chat_message_upsert", "message": message})
        # Kept for older Dashboard clients during the protocol transition.
        broadcast(
            {
                "type": "snapshot_result",
                "capture_id": capture_id,
                "url": f"/captures/{capture_id}",
                "label": label,
                "jpeg": base64.b64encode(jpeg).decode(),  # frontend expects this
            }
        )

    engine.on_turn_event(on_turn_event)
    engine.on_snapshot(on_snapshot)

    # ── WebSocket endpoint ──
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        def _snapshot() -> list[dict[str, Any]]:
            runtime = {**engine.runtime_status(), **sdk_status}
            return [
                {"type": "state", "state": engine._state},
                {
                    "type": "volume",
                    "volume": playback_percent_from_gain(engine._audio.volume)
                    if engine._audio
                    else 80,
                },
                {"type": "wake_word", "enabled": cfg.enable_wake_word},
                {"type": "search", "enabled": cfg.enable_search},
                {"type": "runtime_status", **runtime},
                chat.history_event(),
            ]

        async def _handle_control(client: WebSocket, data: dict[str, Any]) -> None:
            event_type = data.get("type", "")
            if event_type == "get_volume":
                volume = (
                    playback_percent_from_gain(engine._audio.volume)
                    if engine._audio
                    else 80
                )
                await client.send_json({"type": "volume", "volume": volume})
            elif event_type == "set_volume":
                volume = max(0, min(100, int(data.get("volume", 80))))
                if engine._audio:
                    engine._audio.volume = playback_gain_from_percent(volume)
                broadcast({"type": "volume", "volume": volume})
            elif event_type == "get_wake_word":
                await client.send_json(
                    {"type": "wake_word", "enabled": cfg.enable_wake_word}
                )
            elif event_type == "set_wake_word":
                enabled = bool(data.get("enabled", True))
                engine.set_wake_word_enabled(enabled)
                broadcast({"type": "wake_word", "enabled": enabled})
                broadcast(
                    {
                        "type": "state",
                        "state": "wake_listening" if enabled else "listening",
                    }
                )
            elif event_type == "get_state":
                await client.send_json({"type": "state", "state": engine._state})
            elif event_type == "set_search":
                enabled = bool(data.get("enabled", False))
                cfg.enable_search = enabled
                broadcast({"type": "search", "enabled": enabled})
            elif event_type == "get_search":
                await client.send_json({"type": "search", "enabled": cfg.enable_search})
            elif event_type == "get_runtime_status":
                await client.send_json(
                    {"type": "runtime_status", **engine.runtime_status(), **sdk_status}
                )
            elif event_type == "chat_send":
                text = str(data.get("text") or "").strip()
                if text:
                    asyncio.create_task(
                        engine.process_text(
                            text,
                            source="dashboard",
                            client_message_id=str(data.get("client_message_id") or ""),
                        )
                    )
                else:
                    await client.send_json({"type": "error", "message": "消息不能为空"})
            elif event_type == "clear_chat":
                chat.clear()
                engine.clear_conversation()
                broadcast(chat.history_event())
            elif event_type == "browser_location":
                # Browser navigator.geolocation → backend (macOS CoreLocation / GPS)
                lat = float(data.get("lat", 0))
                lon = float(data.get("lon", 0))
                accuracy = data.get("accuracy")  # meters, from browser
                altitude = data.get("altitude")
                heading = data.get("heading")
                speed = data.get("speed")
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    if engine._location is not None:
                        engine._location.set_browser_position(
                            lat,
                            lon,
                            accuracy_m=float(accuracy)
                            if accuracy is not None
                            else None,
                            altitude_m=float(altitude)
                            if altitude is not None
                            else None,
                            heading_deg=float(heading) if heading is not None else None,
                            speed_kmh=float(speed) * 3.6 if speed is not None else None,
                        )
            elif event_type == "set_location":
                lat = float(data.get("lat", 0))
                lon = float(data.get("lon", 0))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    if engine._location is not None:
                        engine._location.set_manual(lat, lon)
                        pos = engine._location.latest_position
                        if pos:
                            await client.send_json(
                                {
                                    "type": "runtime_status",
                                    **engine.runtime_status(),
                                    **sdk_status,
                                }
                            )
                        broadcast({"type": "location_updated", "lat": lat, "lon": lon})
                else:
                    await client.send_json(
                        {
                            "type": "error",
                            "message": "经纬度范围无效（lat: -90~90, lon: -180~180）",
                        }
                    )
            elif event_type == "snapshot":
                asyncio.create_task(
                    engine.process_text(
                        "你看到什么",
                        source="dashboard",
                        client_message_id=str(data.get("client_message_id") or ""),
                    )
                )
            # ── Motion commands ──
            elif event_type == "motion_dance":
                if getattr(engine, "_dance_loop_active", False):
                    broadcast(
                        {"type": "error", "message": "节拍连跳进行中，请先点击停止"}
                    )
                    return
                style = data.get("style", "happy")
                if motion:
                    logger.info("🕹 Dashboard 触发舞蹈: style=%s", style)
                    broadcast(
                        {"type": "motion_status", "action": "dancing", "style": style}
                    )
                    asyncio.create_task(_run_motion(motion.dance(style)))
                else:
                    logger.warning(
                        "motion_dance 事件被忽略: motion 控制器不可用 (style=%s)", style
                    )
            elif event_type == "motion_dance_loop":
                # 无限节拍连跳：播放/停止切换；跳舞期间语音挂起
                action = data.get("action", "")
                if action == "play":
                    if getattr(engine, "_dance_loop_active", False):
                        return
                    if beat_dance is None:
                        await client.send_json(
                            {"type": "error", "message": "节拍连跳未启用"}
                        )
                        return
                    if motion is not None and motion.is_busy:
                        await client.send_json(
                            {
                                "type": "error",
                                "message": "机器人正在执行其他动作，请稍后",
                            }
                        )
                        return
                    reply = await engine.start_beat_dance()
                    broadcast({"type": "dance_loop", "active": True})
                    if reply and "开始跳舞" not in reply:
                        await client.send_json({"type": "error", "message": reply})
                elif action == "stop":
                    await engine.stop_beat_dance()
                    broadcast({"type": "dance_loop", "active": False})
            elif event_type == "motion_nod":
                if motion:
                    broadcast({"type": "motion_status", "action": "nodding"})
                    asyncio.create_task(_run_motion(motion.nod(times=2)))
            elif event_type == "motion_shake_head":
                if motion:
                    broadcast({"type": "motion_status", "action": "shaking"})
                    asyncio.create_task(_run_motion(motion.shake_head(times=2)))
            elif event_type == "motion_wave":
                if motion:
                    broadcast({"type": "motion_status", "action": "waving"})
                    asyncio.create_task(_run_motion(motion.wave_antenna("both")))
            elif event_type == "motion_pose":
                if motion or reachy:
                    action = data.get("action", "")
                    if action == "sleep":
                        if reachy:
                            reachy.goto_sleep()
                        elif motion:
                            asyncio.create_task(_run_motion(motion.sleep()))
                        broadcast({"type": "motion_status", "action": "sleeping"})
                    elif action == "wake_up":
                        if reachy:
                            reachy.wake_up()
                        elif motion:
                            asyncio.create_task(_run_motion(motion.wake_up()))
                        broadcast({"type": "motion_status", "action": "ready"})

        async def _run_motion(coro) -> None:
            try:
                await coro
            except Exception:
                logger.exception("Motion command failed")
            finally:
                broadcast({"type": "motion_status", "action": "idle"})

        try:
            await run_websocket_session(ws, hub, _snapshot, _handle_control)
        except (WebSocketDisconnect, RuntimeError):
            pass

    # ── HTTP endpoints ──
    @app.get("/favicon.ico")
    async def favicon():
        # Return 204 No Content to silence the browser's auto favicon 404
        from fastapi.responses import Response

        return Response(status_code=204)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(
            content=DASHBOARD_HTML,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/status")
    async def status() -> JSONResponse:
        journal_health = engine._journal_fetcher.health()
        return JSONResponse(
            {
                **engine.runtime_status(),
                **sdk_status,
                "camera_device": str(cfg.camera_device),
                "camera_stream": mjpeg.is_active,
                "history_turns": len(engine._conversation_history) // 2,
                "memory_docs": engine._memory.count() if engine._memory else 0,
                "journal": journal_health,
                "journal_complete_ratio": (
                    journal_health["complete"] / journal_health["expected"]
                    if journal_health["expected"]
                    else 0.0
                ),
                "camera_backend": camera_backend.backend_name
                if camera_backend
                else "unavailable",
                "front_frame_age_s": mjpeg.frame_age_s,
                "rear_camera_configured": bool(
                    cfg.ezviz_app_key
                    and cfg.ezviz_app_secret
                    and cfg.ezviz_device_serial
                ),
                "chat_messages": chat.message_count,
                "chat_captures": chat.capture_count,
                "volume": (
                    playback_percent_from_gain(engine._audio.volume)
                    if engine._audio
                    else 0
                ),
                "wake_word_enabled": cfg.enable_wake_word,
                "ws_subscribers": hub.subscriber_count,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Liveness endpoint that reports robot readiness without secrets."""
        try:
            from importlib.metadata import version

            app_version = version("chaihuo-reachy")
        except Exception:
            app_version = "unknown"
        return JSONResponse(
            {
                "status": "ok",
                "version": app_version,
                "started_at": started_at,
                "sdk_connected": bool(sdk_status.get("sdk_connected")),
                "robot_ready": bool(sdk_status.get("robot_ready")),
                "robot_status": sdk_status.get("robot_status", "degraded"),
                "daemon_health": sdk_status.get("daemon_health", "unavailable"),
                "daemon_owner": sdk_status.get("daemon_owner", "none"),
                "daemon_error": sdk_status.get("daemon_error"),
                "audio_backend": (
                    engine._audio.backend_name
                    if engine._audio is not None
                    else "unavailable"
                ),
                "state": engine._state,
            }
        )

    @app.get("/captures/{capture_id}")
    async def capture(capture_id: str) -> Response:
        jpeg = chat.get_capture(capture_id)
        if jpeg is None:
            return Response(status_code=404)
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/camera/stream")
    async def camera_stream() -> StreamingResponse:
        return StreamingResponse(
            generate_mjpeg(mjpeg),
            media_type="multipart/x-mixed-replace; boundary=mjpeg-boundary",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # ── Debug endpoint: text-in / text-out ──────────────────────────
    from pydantic import BaseModel

    class DebugChatRequest(BaseModel):
        text: str
        image_base64: str | None = None  # optional JPEG base64 for vision

    @app.post("/debug/chat")
    async def debug_chat(req: DebugChatRequest) -> JSONResponse:
        """Debug endpoint: send text directly to the LLM pipeline.

        Request:
            POST /debug/chat
            {"text": "今天天气怎么样？", "image_base64": "/9j/4AAQ... (optional)"}

        Returns:
            {"reply": "...", "emotion": "happy", "memory_context": "...",
             "vision_context": "...", "history_turns": 3}
        """
        image_bytes: bytes | None = None
        if req.image_base64:
            try:
                image_bytes = base64.b64decode(req.image_base64)
            except Exception:
                return JSONResponse({"error": "image_base64 解码失败"}, status_code=400)

        result = await engine.process_text(req.text, image_bytes=image_bytes)
        return JSONResponse(
            {
                **result,
                "history_turns": len(engine._conversation_history) // 2,
            }
        )

    # ── Journal sync endpoint ───────────────────────────────────────
    @app.post("/debug/sync-journals")
    async def debug_sync_journals() -> JSONResponse:
        """Manually trigger journal fetch + re-index."""
        from chaihuo_reachy.memory import JournalFetcher

        fetcher = JournalFetcher(
            listing_url=cfg.journal_url,
            cache_dir=cfg.journal_cache_dir,
        )
        results = await fetcher.sync(memory_store=engine._memory)
        return JSONResponse(
            {
                "ok": True,
                "total": len(results),
                "new": sum(1 for r in results if r.get("new")),
                "cached": sum(1 for r in results if not r.get("new")),
                "memory_docs": engine._memory.count() if engine._memory else 0,
            }
        )

    # ── Search toggle endpoint ──────────────────────────────────────
    class SearchToggleRequest(BaseModel):
        enabled: bool

    @app.post("/debug/search/toggle")
    async def debug_search_toggle(req: SearchToggleRequest) -> JSONResponse:
        """Toggle Bailian web search on/off."""
        cfg.enable_search = req.enabled
        broadcast({"type": "search", "enabled": req.enabled})
        return JSONResponse({"ok": True, "search_enabled": cfg.enable_search})

    @app.get("/debug/search/status")
    async def debug_search_status() -> JSONResponse:
        return JSONResponse({"search_enabled": cfg.enable_search})

    # ── Start engine ──
    # Resolve audio and start the engine before advertising a healthy server.
    # Auto mode therefore fails loudly instead of leaving a misleading
    # Dashboard running on the Mac's default microphone.
    await engine.start()

    # ── Periodic audio level broadcast ────────────────────────────────
    async def _broadcast_audio_level() -> None:
        """Poll audio level and camera frame age via WebSocket (~2 Hz)."""
        import math as _math

        while True:
            await asyncio.sleep(0.5)
            if engine._audio is not None:
                rms = engine._audio.capture_rms
                db = round(20.0 * _math.log10(max(rms, 1e-8)), 1)
                broadcast(
                    {
                        "type": "audio_level",
                        "db": db,
                        "rms": round(rms, 4),
                        "frame_age_s": round(mjpeg.frame_age_s, 1)
                        if mjpeg.frame_age_s is not None
                        else None,
                    }
                )

    asyncio.create_task(_broadcast_audio_level())

    # ── Auto-sync journals periodically (background, non-blocking) ──
    async def _auto_sync_journals() -> None:
        from chaihuo_reachy.memory import JournalFetcher
        from chaihuo_reachy.memory.journal_fetcher import journal_sync_lock

        interval = max(0, cfg.journal_auto_sync_interval_minutes) * 60
        while True:
            # Respect the cross-process flock (systemd timer sync, engine's
            # per-answer sync): skip this tick when another sync owns it.
            with journal_sync_lock(cfg.journal_cache_dir) as acquired:
                if acquired:
                    fetcher = JournalFetcher(
                        listing_url=cfg.journal_url,
                        cache_dir=cfg.journal_cache_dir,
                    )
                    try:
                        results = await fetcher.sync(memory_store=engine._memory)
                        new_count = sum(1 for r in results if r.get("new"))
                        if new_count:
                            print(
                                f"  📝 日记自动同步: {new_count} 篇新日记已索引 "
                                f"(共 {len(results)} 篇)"
                            )
                        elif interval > 0:
                            logger.debug("日记自动同步: 无新内容 (%d 篇)", len(results))
                        else:
                            print(f"  📝 日记: {len(results)} 篇已缓存")
                    except Exception:
                        logger.warning("Journal auto-sync failed", exc_info=True)
            if interval <= 0:
                break  # 0 = one-shot only at startup
            await asyncio.sleep(interval)

    asyncio.create_task(_auto_sync_journals())

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=cfg.dashboard_port,
        log_level="warning",
        # A browser WebSocket must not keep Ctrl+C in uvicorn's drain phase
        # indefinitely.  Our own finally chain then sleeps the robot, closes
        # ALSA and terminates the owned SDK daemon.
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)

    uvicorn_handle_exit = server.handle_exit

    def _handle_server_exit(signum: int, frame: Any) -> None:
        # uvicorn temporarily replaces our process SIGINT handler. Arm the
        # same hard backstop at the instant it receives the first signal,
        # rather than waiting for server.serve() to return.
        _arm_shutdown_backstop(owned_daemon_state_file=cfg.daemon_state_file)
        uvicorn_handle_exit(signum, frame)

    server.handle_exit = _handle_server_exit  # type: ignore[method-assign]

    async def _watch_stop_event() -> None:
        if stop_event is None:
            return
        while not stop_event.is_set():
            await asyncio.sleep(0.2)
        server.should_exit = True

    stop_task = asyncio.create_task(_watch_stop_event())
    audio_info = engine._audio.resolved_info if engine._audio else None

    print(f"\n{'=' * 60}")
    print(f"  🚐 皮皮虾 Dashboard 已启动")
    print(f"  🌐 http://localhost:{cfg.dashboard_port}")
    if audio_info:
        if isinstance(audio_info, dict):
            print(
                f"  🎤 {audio_info.get('input_name', '?')} "
                f"({audio_info.get('max_input_channels', '?')}in/"
                f"{audio_info.get('max_output_channels', '?')}out)"
            )
        else:
            print(
                f"  🎤 [{audio_info.input_index}] {audio_info.input_name} "
                f"({audio_info.max_input_channels}in/{audio_info.max_output_channels}out)"
            )
    print(
        f"  🔌 SDK: {'connected' if sdk_status.get('sdk_connected') else 'standalone'}"
    )
    print(
        f"  📷 Reachy Mini Camera ({'MJPEG stream' if mjpeg_ok else 'snapshot only'})"
    )
    print(f"  🔊 音量控制: PCM 增益")
    print(f"  🎤 唤醒词: {'开启' if cfg.enable_wake_word else '关闭'}")
    print(f"{'=' * 60}")
    print(f"  打开浏览器查看 Dashboard，说 '皮皮虾' 唤醒我")
    print(f"  Ctrl+C 退出\n")

    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\n👋 皮皮虾下线！")
    finally:
        _arm_shutdown_backstop()
        if manage_reachy_lifecycle:
            await _sleep_reachy_on_shutdown(reachy, cfg)
        mjpeg.stop()
        await engine.stop()
        if manage_reachy_lifecycle:
            await _close_reachy_runtime(reachy)
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass


# ── CLI voice loop ───────────────────────────────────────────────────────────

# ── Hardware diagnostic ─────────────────────────────────────────────────────


async def run_diagnostic(cfg: Config) -> None:
    import sounddevice as sd
    from chaihuo_reachy.audio import AudioDeviceResolutionError, resolve_audio_device

    print("=" * 60)
    print("  🔧 柴火基地车 Reachy Mini 硬件诊断")
    print("=" * 60)

    print("\n📋 音频设备:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        marker = " ← Reachy Mini!" if "reachy" in d["name"].lower() else ""
        print(
            f"  [{i}] {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']}){marker}"
        )

    try:
        resolved = resolve_audio_device(
            cfg.audio_device,
            devices=devices,
            default_device=sd.default.device,
        )
    except AudioDeviceResolutionError as exc:
        resolved = None
        print(f"\n❌ 音频设备解析失败: {exc}")

    if resolved is not None:
        print(
            f"\n✅ 实际音频: input=[{resolved.input_index}] {resolved.input_name} "
            f"({resolved.max_input_channels}ch), output=[{resolved.output_index}] "
            f"{resolved.output_name} ({resolved.max_output_channels}ch), "
            f"backend={resolved.backend}"
        )
        print("🎤 测试麦克风...")
        try:
            audio = sd.rec(
                int(16000 * 2),
                samplerate=16000,
                channels=1,
                device=resolved.input_index,
                dtype="int16",
            )
            sd.wait()
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            print(f"   ✅ RMS={rms:.1f}" + (" (有声音!)" if rms > 500 else ""))
        except Exception as e:
            print(f"   ❌ {e}")
        print("🔊 测试扬声器...")
        try:
            t = np.linspace(0, 0.3, int(16000 * 0.3), False)
            beep = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
            sd.play(beep, samplerate=16000, device=resolved.output_index)
            sd.wait()
            print("   ✅ 应该听到'哔'一声")
        except Exception as e:
            print(f"   ❌ {e}")
    else:
        print("   提示: 本地 Mac 音频必须显式设置 REACHY_AUDIO_DEVICE=default")

    print("\n📷 摄像头:")
    found = False
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            if ret and frame is not None:
                h, w = frame.shape[:2]
                m = " ← Reachy Mini" if abs(fps - 15) < 3 else ""
                if m:
                    found = True
                print(f"  [{i}] {w}x{h} @ {fps:.0f}fps{m}")
            cap.release()
    print(f"  {'✅' if found else '⚠️'} Reachy Mini 摄像头")

    print("\n🌐 百炼 API:")
    try:
        from chaihuo_reachy.bailian import BailianLLMClient

        async with BailianLLMClient(cfg) as llm:
            async for _ in llm.chat_stream([{"role": "user", "content": "OK"}]):
                pass
        print("   ✅ LLM OK")
    except Exception as e:
        print(f"   ❌ LLM: {e}")

    try:
        from chaihuo_reachy.bailian import BailianTTSClient

        chunks: list[bytes] = []
        tts = BailianTTSClient(cfg, on_audio=lambda b, sr: chunks.append(b))
        await tts.open()
        await tts.synthesize("测试")
        await tts.close()
        print(f"   ✅ TTS OK ({len(chunks)} chunks)")
    except Exception as e:
        print(f"   ❌ TTS: {e}")

    print("\n🗣️ 本地唤醒词 (KWS):")
    if cfg.enable_wake_word and cfg.wake_engine == "local":
        from chaihuo_reachy.wake_word import WakeWordDetector, WakeWordUnavailableError

        try:
            detector = WakeWordDetector(cfg)
            detector.self_check()
            print(
                f"   ✅ KWS 就绪: model={cfg.kws_model_dir} "
                f"threshold={cfg.kws_threshold} score={cfg.kws_score}"
            )
            print("   说 '皮皮虾' 测试唤醒")
        except WakeWordUnavailableError as e:
            print(f"   ⚠️ 本地 KWS 不可用(将回退云端唤醒): {e}")
        except Exception as e:
            print(f"   ⚠️ 本地 KWS 异常(将回退云端唤醒): {type(e).__name__}: {e}")
    elif cfg.wake_engine == "cloud":
        print("   ℹ️ 云端 ASR 文本匹配唤醒 (REACHY_WAKE_ENGINE=cloud)")
    else:
        print("   ℹ️ 唤醒词已禁用")

    print(f"\n{'=' * 60}")
    print("  诊断完成！uv run chaihuo-reachy dashboard")
    print(f"{'=' * 60}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def _daemon_hosts(cfg: Config) -> list[str]:
    """Return the configured daemon target without probing unrelated daemons."""
    configured = str(cfg.daemon_host or "").strip()
    return [configured or "localhost"]


async def _daemon_port_is_occupied(cfg: Config) -> bool:
    """Detect a local listener without inspecting or terminating its process."""
    if cfg.daemon_host not in {"", "localhost", "127.0.0.1", "::1"}:
        return False
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("localhost", cfg.daemon_port), timeout=0.3
        )
        del reader
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def _connect_reachy_once(
    reachy_cls: type,
    cfg: Config,
    host: str,
):
    """Connect without blocking the Dashboard event loop."""
    connection_mode = (
        "localhost_only" if host in {"localhost", "127.0.0.1", "::1"} else "network"
    )
    return await asyncio.to_thread(
        reachy_cls,
        host=host,
        port=cfg.daemon_port,
        connection_mode=connection_mode,
        media_backend=("local" if cfg.media_backend != "no_media" else "no_media"),
        spawn_daemon=False,
        use_sim=cfg.daemon_simulation,
        timeout=_DAEMON_CONNECT_TIMEOUT_S,
    )


def _resolve_daemon_serial_port(cfg: Config) -> str:
    """Resolve an explicitly configured controller, with safe USB recovery."""
    configured = str(cfg.daemon_serial_port or "").strip()
    if cfg.daemon_simulation:
        return configured
    if configured and Path(configured).exists():
        return configured
    candidates = sorted(Path("/dev").glob("cu.usbmodem*"))
    if len(candidates) == 1:
        resolved = str(candidates[0])
        logger.warning("配置串口不可用，自动恢复唯一 USB 串口: %s", resolved)
        return resolved
    if configured:
        if len(candidates) > 1:
            raise RuntimeError(
                f"配置串口不存在，发现多个候选串口: {', '.join(map(str, candidates))}"
            )
        raise RuntimeError(
            f"配置串口不存在且未发现 /dev/cu.usbmodem* 设备: {configured}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"未配置串口，发现多个候选串口: {', '.join(map(str, candidates))}"
        )
    if not candidates:
        raise RuntimeError("未配置串口且未发现 /dev/cu.usbmodem* 设备")
    return str(candidates[0])


def _spawn_sdk_daemon_process(cfg: Config, resolved_serial_port: str | None = None):
    """Start the SDK daemon and retain an owned process handle for cleanup.

    Hardware mode is the default. MuJoCo simulation must be explicitly opted
    into because the plain Reachy Mini package does not install its optional
    ``mujoco`` dependency.
    """
    import shutil
    import subprocess

    executable = shutil.which("reachy-mini-daemon")
    if not executable:
        # Non-interactive SSH/nohup launches often do not inherit the venv's
        # ``bin`` directory in PATH even though this application itself is
        # running from that venv.  The SDK daemon is installed beside the
        # current Python interpreter, so resolve that deterministic sibling
        # before declaring the runtime unavailable.
        # Do not resolve the Python symlink: venv/bin/python commonly points
        # to /usr/bin/python, while the console script we need remains in the
        # original venv/bin directory.
        venv_daemon = Path(sys.executable).with_name("reachy-mini-daemon")
        if venv_daemon.is_file() and os.access(venv_daemon, os.X_OK):
            executable = str(venv_daemon)
    if not executable:
        raise RuntimeError("reachy-mini-daemon executable was not found")
    command = [executable]
    # Validation/recovery is performed by the lifecycle flow before this
    # low-level launcher is called.  Keeping the launcher deterministic also
    # makes it usable by deployment tooling that already resolved a device.
    serial_port = resolved_serial_port
    if serial_port is None and not cfg.daemon_simulation:
        serial_port = cfg.daemon_serial_port
    if serial_port:
        command.extend(["--serialport", serial_port])
    if cfg.daemon_simulation:
        command.append("--sim")
    if cfg.media_backend == "no_media":
        command.append("--no-media")
    process = subprocess.Popen(command, start_new_session=True)
    if getattr(process, "pid", None):
        daemon_runtime.write_state(
            cfg.daemon_state_file,
            daemon_runtime.make_state(
                process,
                command,
                host=cfg.daemon_host,
                port=cfg.daemon_port,
                serial_port=serial_port or "",
                simulation=cfg.daemon_simulation,
                no_media=cfg.media_backend == "no_media",
            ),
        )
    return process


async def _find_reachy_daemon(
    reachy_cls: type,
    cfg: Config,
    *,
    timeout_s: float,
    daemon_process: Any | None = None,
):
    """Poll a daemon that may still be loading motors and media pipelines."""
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout_s)
    attempt = 0
    while True:
        if daemon_process is not None and daemon_process.poll() is not None:
            raise _DaemonStartupError(
                f"reachy-mini-daemon exited with code {daemon_process.returncode}"
            )
        attempt += 1
        for host in _daemon_hosts(cfg):
            try:
                backend_error = await _daemon_backend_error(host, cfg.daemon_port, cfg)
                if backend_error:
                    # A daemon accepts SDK connections before its asynchronous
                    # motor wake-up completes.  For the exact process we just
                    # started, ``ready=false`` is an initialization state, not
                    # a terminal fault.  Existing daemons are still rejected
                    # immediately so Dashboard can safely enter degraded mode.
                    if not _daemon_is_initializing(backend_error, daemon_process):
                        raise _DaemonStartupError(backend_error)
                    logger.info("Reachy daemon 正在初始化硬件，等待 ready=true...")
                    continue
                # While a daemon is waking, repeatedly constructing an SDK
                # client also constructs local GStreamer media pipelines.  Do
                # the inexpensive HTTP readiness poll first and open SDK
                # media only once the daemon reports a usable backend.
                reachy = await _connect_reachy_once(reachy_cls, cfg, host)
                logger.info(
                    "✅ Reachy Mini daemon 已就绪 (host=%s, 尝试=%d)",
                    host,
                    attempt,
                )
                return reachy
            except Exception as exc:
                logger.debug("Daemon 尚未就绪 (%s): %s", host, exc)
                backend_error = await _daemon_backend_error(
                    host,
                    cfg.daemon_port,
                    cfg,
                )
                if backend_error:
                    if not _daemon_is_initializing(backend_error, daemon_process):
                        raise _DaemonStartupError(backend_error)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(_DAEMON_POLL_INTERVAL_S, remaining))


def _daemon_is_initializing(error: str, daemon_process: Any | None) -> bool:
    """Whether a just-spawned daemon should receive another readiness poll."""
    return (
        daemon_process is not None
        and daemon_process.poll() is None
        and "backend_status.ready=false" in error
    )


def _backend_reports_live_control(backend: dict[str, Any]) -> bool:
    """Accept the SDK 1.9 false-negative ready flag only with live control."""
    stats = backend.get("control_loop_stats")
    return (
        str(backend.get("motor_control_mode") or "").lower() == "enabled"
        and isinstance(stats, dict)
        and int(stats.get("nb_error", 1)) == 0
        and float(stats.get("mean_control_loop_frequency", 0.0)) >= 20.0
    )


async def _daemon_backend_error(host: str, port: int, cfg: Config | None = None) -> str:
    """Validate daemon HTTP health when its status endpoint is reachable.

    A Reachy client socket can be accepted while the motor backend is still in
    an error state.  The status payload is therefore the authority for robot
    readiness, not merely a successful TCP/SDK connection.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=0.5) as client:
            response = await client.get(f"http://{host}:{port}/api/daemon/status")
        if response.status_code != 200:
            return f"daemon 健康检查返回 HTTP {response.status_code}"
        payload = response.json()
        if not isinstance(payload, dict):
            return "daemon 健康检查返回了无效数据"
        state = str(payload.get("state") or "").lower()
        if state != "running":
            return str(payload.get("error") or "Reachy daemon backend failed")
        if payload.get("error"):
            return str(payload["error"])
        backend = payload.get("backend_status")
        if not isinstance(backend, dict):
            return "daemon 未返回 backend_status，无法确认机器人已就绪"
        if backend.get("error"):
            return str(backend["error"])
        if backend.get("ready") is not True:
            # reachy-mini 1.9 on physical hardware can keep ``ready=false``
            # after a successful wake-up.  A running, error-free controller
            # with a live 20+ Hz loop is a stronger safety signal than that
            # stale flag.  Do not endlessly reopen media pipelines waiting for
            # a value the daemon never updates.
            if _backend_reports_live_control(backend):
                logger.warning(
                    "daemon backend ready=false，但电机控制环正常运行；按兼容模式继续"
                )
                return ""
            return "机器人硬件未就绪 (backend_status.ready=false)"
        if cfg and cfg.daemon_simulation:
            if payload.get("simulation_enabled") is not True:
                return "daemon 非模拟模式，与 REACHY_DAEMON_SIMULATION=true 不匹配"
        elif cfg and payload.get("simulation_enabled") is True:
            return "daemon 为模拟模式，与物理硬件配置不匹配"
    except Exception:
        return ""
    return ""


async def _wake_up_reachy(reachy: Any, *, attempts: int = 3) -> bool:
    """Wake the robot with bounded retries after daemon readiness."""
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.to_thread(reachy.wake_up)
            logger.info("✅ 皮皮虾已就绪 — 头部归位，天线展开")
            return True
        except Exception:
            logger.warning(
                "wake_up 第 %d/%d 次失败",
                attempt,
                attempts,
                exc_info=True,
            )
            if attempt < attempts:
                await asyncio.sleep(1.0)
    return False


async def _sleep_reachy_on_shutdown(reachy: Any | None, cfg: Config) -> bool:
    """Put the robot in its safe resting pose before releasing the daemon."""
    if reachy is None or not cfg.auto_sleep:
        return True
    if getattr(reachy, "_chaihuo_robot_slept", False) or getattr(
        reachy, "_chaihuo_robot_sleeping", False
    ):
        return True
    daemon_process = getattr(reachy, "_chaihuo_daemon_process", None)
    if daemon_process is not None and daemon_process.poll() is None:
        # An owned daemon runs goto_sleep as part of its SIGTERM shutdown.
        # Calling it through the client as well produces a second stop cue.
        setattr(reachy, "_chaihuo_robot_slept", True)
        logger.info("本项目 daemon 将在退出时让皮皮虾休眠")
        return True
    # The two Dashboard cleanup layers can overlap during Ctrl+C.  Claim the
    # operation before its first await so only one command reaches the robot.
    setattr(reachy, "_chaihuo_robot_sleeping", True)
    logger.info("🛌 Ctrl+C/退出：正在让皮皮虾休眠...")
    try:
        await asyncio.wait_for(
            asyncio.to_thread(reachy.goto_sleep),
            timeout=10.0,
        )
        setattr(reachy, "_chaihuo_robot_slept", True)
        logger.info("😴 皮皮虾已休眠")
        return True
    except Exception:
        logger.exception("退出时 goto_sleep 失败")
        return False
    finally:
        setattr(reachy, "_chaihuo_robot_sleeping", False)


def _daemon_owner(cfg: Config) -> str:
    """Classify daemon ownership from the persisted, validated manifest."""
    state = daemon_runtime.read_state(cfg.daemon_state_file)
    if state is None:
        return "none"
    if daemon_runtime.owned_process(state):
        return "owned"
    # A state file is not authority by itself.  Remove stale records so a
    # future ``spawn`` can proceed; a mismatched live process is left alone.
    daemon_runtime.remove_state(cfg.daemon_state_file)
    return "none"


async def _recover_owned_daemon(cfg: Config) -> bool:
    """Request graceful shutdown of only a manifest-verified daemon."""
    if _daemon_owner(cfg) != "owned":
        return False
    stopped = await asyncio.to_thread(
        daemon_runtime.terminate_owned_state,
        cfg.daemon_state_file,
    )
    if stopped:
        logger.info("已回收本项目记录的 reachy-mini-daemon")
        await asyncio.sleep(0.5)
    return stopped


def _degraded_sdk_status(
    cfg: Config, reason: str, *, owner: str | None = None
) -> dict[str, Any]:
    return {
        "sdk_connected": False,
        "mode": "standalone_degraded",
        "robot_ready": False,
        "robot_status": "degraded",
        "daemon_health": "unavailable",
        "daemon_owner": owner or _daemon_owner(cfg),
        "daemon_error": reason or "daemon 不可用",
        "daemon_host": cfg.daemon_host,
        "daemon_port": cfg.daemon_port,
        "serial_port": cfg.daemon_serial_port or None,
        "media_backend": "direct",
    }


async def _terminate_owned_daemon(
    process: Any | None,
    *,
    state_file: str | None = None,
) -> None:
    """Terminate only the exact daemon process spawned by this application."""
    if process is None or process.poll() is not None:
        state = daemon_runtime.read_state(state_file) if state_file else None
        if state_file and (
            state is None or state.get("pid") == getattr(process, "pid", None)
        ):
            daemon_runtime.remove_state(state_file)
        return
    logger.info("正在停止本次启动的 reachy-mini-daemon (PID %s)...", process.pid)
    process.terminate()
    try:
        await asyncio.to_thread(process.wait, 8.0)
    except Exception:
        logger.warning("daemon 未在 8 秒内退出，发送强制终止")
        process.kill()
        await asyncio.to_thread(process.wait)
    logger.info("✅ reachy-mini-daemon 已停止")
    if state_file:
        state = daemon_runtime.read_state(state_file)
        if state is None or state.get("pid") == getattr(process, "pid", None):
            daemon_runtime.remove_state(state_file)


async def _close_reachy_runtime(reachy: Any | None) -> None:
    """Close client media and stop an owned daemon after the robot sleeps."""
    if reachy is None:
        return
    if getattr(reachy, "_chaihuo_runtime_closed", False):
        return
    daemon_process = getattr(reachy, "_chaihuo_daemon_process", None)
    media_manager = getattr(reachy, "media_manager", None)
    if media_manager is not None and daemon_process is None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(media_manager.close),
                timeout=5.0,
            )
        except Exception:
            logger.warning("SDK media manager 关闭超时或失败", exc_info=True)
    elif daemon_process is not None:
        logger.debug("本项目 daemon 负责关闭 SDK media manager")
    await _terminate_owned_daemon(
        daemon_process,
        state_file=getattr(reachy, "_chaihuo_daemon_state_file", None),
    )
    setattr(reachy, "_chaihuo_runtime_closed", True)


async def _try_connect_daemon_impl(cfg: Config) -> tuple:
    """Try to connect to the Reachy Mini daemon and set up SDK backends.

    ``auto`` reuses a healthy daemon and starts a local one only when absent.
    ``connect`` never starts a process.  ``spawn`` may first recover only a
    daemon that is verified by this project's ownership manifest.

    Returns:
        (reachy, audio_backend, camera_backend, motion, sdk_status)
        All None if daemon is unreachable.
    """
    from reachy_mini import ReachyMini

    mode = str(cfg.daemon_mode or "auto").strip().lower()
    if mode not in {"auto", "connect", "spawn"}:
        return (
            None,
            None,
            None,
            None,
            None,
            _degraded_sdk_status(cfg, f"无效的 REACHY_DAEMON_MODE: {cfg.daemon_mode}"),
        )
    from chaihuo_reachy.beat_dance import BeatDanceController

    # ── Strategy 1: connect without assuming ownership ───────────────
    terminal_error = ""
    daemon_process = None
    try:
        reachy = await _find_reachy_daemon(ReachyMini, cfg, timeout_s=0)
    except _DaemonStartupError as exc:
        reachy = None
        terminal_error = str(exc)
    if reachy is None and mode == "spawn":
        # Explicit recovery mode never touches a daemon not recorded by us.
        await _recover_owned_daemon(cfg)
        terminal_error = ""

    can_spawn_locally = cfg.daemon_host in {"", "localhost", "127.0.0.1", "::1"}
    if (
        reachy is None
        and not terminal_error
        and mode == "auto"
        and can_spawn_locally
        and await _daemon_port_is_occupied(cfg)
    ):
        # The port is held by a daemon we could not connect to.  If the
        # ownership manifest still validates it (a leftover from a previous
        # run whose shutdown was aborted, or one whose motor link died),
        # reclaim it and fall through to a fresh spawn — otherwise the
        # robot would stay dead forever behind an unhealthy zombie.
        if _daemon_owner(cfg) == "owned":
            logger.warning(
                "检测到本项目遗留的 reachy-mini-daemon（PID 来自 %s），回收后重新拉起",
                cfg.daemon_state_file,
            )
            await _recover_owned_daemon(cfg)
        else:
            terminal_error = "本机 daemon 端口已被外部或异常进程占用，auto 模式不会接管"
    if (
        reachy is None
        and not terminal_error
        and mode != "connect"
        and can_spawn_locally
    ):
        # ── Strategy 2: spawn once, then wait for strict readiness ──
        logger.info("未发现健康的本机 daemon，正在拉起 Reachy Mini 进程...")
        try:
            serial_port = await asyncio.to_thread(_resolve_daemon_serial_port, cfg)
            # Persist the runtime-selected port in the in-memory status so
            # operators can see which controller was actually used.
            if serial_port:
                cfg.daemon_serial_port = serial_port
            daemon_process = await asyncio.to_thread(
                _spawn_sdk_daemon_process, cfg, serial_port
            )
            # The public wrapper uses this exact handle to finish daemon
            # shutdown if Ctrl+C arrives before this function returns it.
            setattr(cfg, "_chaihuo_starting_daemon_process", daemon_process)
        except Exception as exc:
            terminal_error = str(exc)
            logger.error("无法拉起 SDK daemon：%s", terminal_error)
        if daemon_process is not None:
            logger.info(
                "等待 daemon 完成初始化（最多 %.0f 秒）...",
                _DAEMON_STARTUP_TIMEOUT_S,
            )
            try:
                reachy = await _find_reachy_daemon(
                    ReachyMini,
                    cfg,
                    timeout_s=_DAEMON_STARTUP_TIMEOUT_S,
                    daemon_process=daemon_process,
                )
            except _DaemonStartupError as exc:
                terminal_error = str(exc)
                logger.error("Reachy daemon 硬件初始化失败：%s", terminal_error)

    if reachy is None:
        await _terminate_owned_daemon(daemon_process, state_file=cfg.daemon_state_file)
        setattr(cfg, "_chaihuo_starting_daemon_process", None)
        if not terminal_error:
            if mode == "connect":
                terminal_error = "connect 模式下未发现健康 daemon"
            elif not can_spawn_locally:
                terminal_error = (
                    "远程 daemon 不可用；auto/spawn 不会在远程地址拉起本地进程"
                )
            else:
                terminal_error = "未发现健康 daemon"
        logger.error("Daemon 不可用 — Dashboard 将以降级模式启动：%s", terminal_error)
        # 6-tuple to match _start_dashboard/_start_voice_loop unpacking
        # (beat_dance slot was missing → ValueError on daemon-less boot).
        return (None, None, None, None, None, _degraded_sdk_status(cfg, terminal_error))

    # Explicit ownership marker: only a daemon started by this invocation
    # may be terminated when Ctrl+C is received.
    setattr(reachy, "_chaihuo_daemon_process", daemon_process)
    if daemon_process is not None:
        setattr(reachy, "_chaihuo_daemon_state_file", cfg.daemon_state_file)

    # ── Wake up the robot ──────────────────────────────────────────
    robot_ready = True
    if cfg.auto_wake_up:
        logger.info("🤖 皮皮虾正在站起来...")
        robot_ready = await _wake_up_reachy(reachy)
        if not robot_ready:
            logger.warning("⚠️  机器人未能站起；Dashboard 继续启动并显示未就绪")

    # ── Create SDK backends ────────────────────────────────────────
    from chaihuo_reachy.backends.factory import (
        create_audio_backend,
        create_camera_backend,
    )
    from chaihuo_reachy.motion import MotionController

    try:
        audio_backend = create_audio_backend(cfg, reachy.media_manager)
        camera_backend = create_camera_backend(cfg, reachy.media_manager)
        motion = MotionController(reachy, cfg) if cfg.dance_enabled else None
        beat_dance = (
            BeatDanceController(reachy, cfg)
            if cfg.dance_enabled and cfg.beat_dance_enabled
            else None
        )

        if cfg.wobbling_enabled and motion:
            motion.enable_wobbling()
    except Exception:
        await _sleep_reachy_on_shutdown(reachy, cfg)
        await _close_reachy_runtime(reachy)
        raise

    sdk_status = {
        "sdk_connected": True,
        "mode": "daemon_connected",
        "media_backend": "sdk_gstreamer"
        if cfg.media_backend != "no_media"
        else "direct",
        "robot_ready": robot_ready,
        "robot_status": "ready" if robot_ready else "degraded",
        "daemon_health": "healthy",
        "daemon_owner": (
            "owned"
            if daemon_process is not None or _daemon_owner(cfg) == "owned"
            else "external"
        ),
        "daemon_error": None,
        "daemon_host": getattr(
            getattr(reachy, "client", None), "host", cfg.daemon_host
        ),
        "daemon_port": cfg.daemon_port,
        "serial_port": cfg.daemon_serial_port or None,
    }
    logger.info("✅ Daemon 连接成功: %s", sdk_status)
    return (reachy, audio_backend, camera_backend, motion, beat_dance, sdk_status)


async def _try_connect_daemon(cfg: Config) -> tuple:
    """Connect with cancellation-safe cleanup for a daemon still starting."""
    try:
        return await _try_connect_daemon_impl(cfg)
    except asyncio.CancelledError:
        daemon_process = getattr(cfg, "_chaihuo_starting_daemon_process", None)
        if daemon_process is not None:
            logger.info("收到 Ctrl+C，正在停止本次启动的 daemon，使机器人坐下...")
            try:
                await _terminate_owned_daemon(
                    daemon_process,
                    state_file=cfg.daemon_state_file,
                )
            except asyncio.CancelledError:
                # A second Ctrl+C must not prevent the safe signal from being
                # delivered, even though it may skip waiting for completion.
                try:
                    daemon_process.terminate()
                except Exception:
                    logger.debug("Ctrl+C 后 daemon 终止信号发送失败", exc_info=True)
                raise
        raise


# ── Graceful shutdown guarantees ────────────────────────────────────────
#
# Ctrl+C must never leave the XMOS sound card or the dashboard port
# occupied, or the next launch fails with "device busy" / "port in use".
# Three layers: the graceful finally chains below, a shutdown backstop
# daemon thread (auto force-exit after N seconds), and a second Ctrl+C
# that force-exits immediately.

_SHUTDOWN_BACKSTOP_ARMED = threading.Event()
_GOT_FIRST_SIGINT = threading.Event()


def _arm_shutdown_backstop(
    delay_s: float = 25.0,
    *,
    force_exit=os._exit,
    owned_daemon_state_file: str | None = None,
) -> None:
    """Guarantee Ctrl+C releases the audio device and dashboard port.

    Armed at the top of every shutdown finally.  If the graceful cleanup
    chain hangs (wedged ALSA read, stuck WebSocket), the daemon thread
    force-exits the process — the OS then releases the XMOS sound card
    and the port, so the next launch always succeeds.  Idempotent.
    """
    if _SHUTDOWN_BACKSTOP_ARMED.is_set():
        return
    _SHUTDOWN_BACKSTOP_ARMED.set()

    def _force() -> None:
        time.sleep(delay_s)
        logger.error("🛑 资源清理超时（%.0fs），强制退出以释放音频设备与端口", delay_s)
        if owned_daemon_state_file:
            try:
                daemon_runtime.terminate_owned_state(owned_daemon_state_file)
            except Exception:
                logger.exception("强制退出前回收 owned daemon 失败")
        force_exit(1)

    threading.Thread(target=_force, name="shutdown-backstop", daemon=True).start()


def _install_sigint_handler(cfg: Config) -> None:
    """First Ctrl+C → graceful shutdown; second Ctrl+C → force exit.

    The graceful finally chains sleep the robot, close the ALSA PCMs and
    stop the owned daemon.  If that path hangs, pressing Ctrl+C again
    force-exits so the sound card / port are never left occupied.
    """

    def _on_sigint(signum: int, frame: Any) -> None:
        if _GOT_FIRST_SIGINT.is_set():
            logger.error("🛑 再次 Ctrl+C，强制退出（释放音频设备与端口）")
            # The graceful finally chain is being skipped, so reclaim the
            # owned daemon here — otherwise it survives as an orphan that
            # blocks the serial port and the next launch's daemon port.
            try:
                daemon_runtime.terminate_owned_state(cfg.daemon_state_file)
            except Exception:
                logger.exception("强制退出前回收 owned daemon 失败")
            os._exit(128 + signum)
        _GOT_FIRST_SIGINT.set()
        _arm_shutdown_backstop(owned_daemon_state_file=cfg.daemon_state_file)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="🚐 柴火基地车 Reachy Mini 智能助手")
    parser.add_argument(
        "command",
        nargs="?",
        default="dashboard",
        choices=["run", "dashboard", "test", "index-journals", "sync-journals"],
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-wake-word", action="store_true")
    parser.add_argument(
        "--wake-engine",
        choices=["local", "cloud", "off"],
        help="唤醒词引擎: local=本地KWS, cloud=云端文本匹配, off=关闭",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="跳过 daemon 连接，直接用本地音频/摄像头",
    )
    parser.add_argument("--target", choices=["mac", "jetson"])
    return parser


def main() -> None:
    args = build_parser().parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    if args.no_wake_word:
        cfg.enable_wake_word = False
    if args.wake_engine == "off":
        cfg.enable_wake_word = False
    elif args.wake_engine:
        cfg.wake_engine = args.wake_engine
    if args.target:
        cfg.target = args.target

    if cfg.target == "jetson" or os.environ.get("REACHY_TARGET") == "jetson":
        if cfg.camera_device == "auto":
            # Video node numbers are unstable across reboots (video0/video1);
            # resolve by name via the sysfs detector instead of hardcoding.
            from chaihuo_reachy.camera import find_reachy_camera

            cfg.camera_device = find_reachy_camera("auto")
    elif cfg.target == "mac" and cfg.daemon_host == "reachy-mini.local":
        # macOS Lite: daemon runs locally (USB-connected), not on the robot.
        # Only override the default; explicit REACHY_DAEMON_HOST wins.
        cfg.daemon_host = "localhost"

    if args.command == "index-journals":
        asyncio.run(_index_all_journals(cfg))
        return

    if args.command == "sync-journals":
        # Unattended incremental sync (systemd timer): no API key needed.
        sys.exit(asyncio.run(_sync_journals_incremental(cfg)))

    if not cfg.bailian_api_key:
        print("❌ BAILIAN_API_KEY 未设置", file=sys.stderr)
        sys.exit(1)

    # Always use the Reachy Mini SDK daemon.  The optional desktop Control
    # application is deliberately not part of this process lifecycle.
    # Only skip daemon handling when --standalone is explicit.
    standalone = getattr(args, "standalone", False)
    _install_sigint_handler(cfg)

    if args.command == "test":
        asyncio.run(run_diagnostic(cfg))
    elif args.command == "run":
        asyncio.run(_start_voice_loop(cfg, standalone=standalone))
    else:
        asyncio.run(_start_dashboard(cfg, standalone=standalone))


async def _index_all_journals(cfg: Config) -> None:
    """Download and index every entry currently present in the official index."""
    from chaihuo_reachy.memory import JournalFetcher, MemoryStore

    store = MemoryStore(
        persist_dir=cfg.chroma_persist_dir,
        journal_dir=cfg.journal_cache_dir,
    )
    fetcher = JournalFetcher(
        listing_url=cfg.journal_url,
        cache_dir=cfg.journal_cache_dir,
    )
    results = await fetcher.sync(memory_store=store, refresh_all=True)
    health = fetcher.health()
    if health["expected"] != health["complete"]:
        logger.warning(
            "日记不完整：官方 %d 篇，完整 %d 篇（%d 篇不可访问）",
            health["expected"],
            health["complete"],
            health["expected"] - health["complete"],
        )
    print(
        f"✅ 官方目录 {health['expected']} 篇，完整保存 {health['complete']} 篇，"
        f"索引 {store.chunk_count()} 个正文分段"
    )


async def _sync_journals_incremental(cfg: Config) -> int:
    """Incremental journal sync for unattended use; returns process exit code.

    Exit 0: sync succeeded, or another sync is already running (skip), or
    the listing is available but some entries are inaccessible (e.g. Yuque
    401 on private entries) — the corpus stays partially cached and the
    next tick retries them.  Exit 1: listing unavailable / hard error.
    """
    from chaihuo_reachy.memory import JournalFetcher, MemoryStore
    from chaihuo_reachy.memory.journal_fetcher import journal_sync_lock

    cache_dir = Path(cfg.journal_cache_dir)
    with journal_sync_lock(cache_dir) as acquired:
        if not acquired:
            print("another sync in progress, skip")
            return 0
        store = MemoryStore(
            persist_dir=cfg.chroma_persist_dir,
            journal_dir=str(cache_dir),
        )
        fetcher = JournalFetcher(
            listing_url=cfg.journal_url,
            cache_dir=cache_dir,
        )
        try:
            results = await fetcher.sync(memory_store=store)
        except Exception as exc:
            print(f"❌ 日记同步失败: {exc}", file=sys.stderr)
            return 1
        health = fetcher.health()
        new_count = sum(1 for r in results if r.get("new"))
        changed_count = sum(1 for r in results if r.get("changed"))
        if health["complete"] < health["expected"]:
            print(
                f"⚠️ 日记不完整: 官方 {health['expected']} 篇, "
                f"完整 {health['complete']} 篇 "
                f"({health['expected'] - health['complete']} 篇暂不可访问, 下轮重试)"
            )
        print(
            f"✅ 日记同步完成: 新增 {new_count}, 更新 {changed_count}; "
            f"官方 {health['expected']} 篇, 完整 {health['complete']} 篇"
        )
        return 0


async def _start_dashboard(cfg: Config, *, standalone: bool = False) -> None:
    """Start dashboard — try daemon connection first, fall back to standalone."""
    if standalone:
        reachy = audio_backend = camera_backend = motion = beat_dance = None
        sdk_status = {"sdk_connected": False, "mode": "standalone"}
    else:
        (
            reachy,
            audio_backend,
            camera_backend,
            motion,
            beat_dance,
            sdk_status,
        ) = await _try_connect_daemon(cfg)
    # When running without the robot (standalone or daemon unavailable),
    # switch to the default Mac/PC audio device instead of looking for
    # the Reachy Mini sound card.
    if audio_backend is None and cfg.audio_device in (None, "auto"):
        cfg.audio_device = "default"
    try:
        await run_dashboard(
            cfg,
            sdk_status=sdk_status or {"sdk_connected": False, "mode": "standalone"},
            audio_backend=audio_backend,
            camera_backend=camera_backend,
            reachy=reachy,
            motion=motion,
            beat_dance=beat_dance,
            manage_reachy_lifecycle=True,
        )
    finally:
        _arm_shutdown_backstop()
        # Covers failures before run_dashboard reaches its own server finally
        # (audio/camera/memory initialisation), while the idempotency markers
        # make the normal Ctrl+C path a no-op here.
        await _sleep_reachy_on_shutdown(reachy, cfg)
        await _close_reachy_runtime(reachy)


async def _start_voice_loop(cfg: Config, *, standalone: bool = False) -> None:
    """Start voice loop — try daemon connection, fall back to standalone."""
    if standalone:
        reachy = audio_backend = camera_backend = motion = None
    else:
        reachy, audio_backend, camera_backend, motion, _, _ = await _try_connect_daemon(
            cfg
        )
    # When running without the robot, switch to default system audio.
    if audio_backend is None and cfg.audio_device in (None, "auto"):
        cfg.audio_device = "default"
    engine = ConversationEngine(
        cfg,
        audio_backend=audio_backend,
        camera_backend=camera_backend,
        motion=motion,
    )
    await _run_voice_loop_engine(engine, cfg, reachy=reachy)


async def _run_voice_loop_engine(
    engine: ConversationEngine,
    cfg: Config,
    *,
    reachy: Any | None = None,
) -> None:
    """Run the terminal voice loop with the given engine."""

    def on_state(state: str) -> None:
        emoji = {
            "idle": "💤",
            "listening": "🎤",
            "thinking": "🤔",
            "speaking": "🔊",
            "wake_listening": "👂",
        }
        print(f"\n  {emoji.get(state, '?')} [{state}]", end=" ", flush=True)

    def on_transcript(text: str, is_final: bool) -> None:
        if is_final:
            print(f"\n  📝 你说: {text}")
        else:
            print(f"\r  🎤 ...{text[-30:]}", end="", flush=True)

    def on_llm_token(token: str) -> None:
        print(token, end="", flush=True)

    engine.on_state_change(on_state)
    engine.on_transcript(on_transcript)
    engine.on_llm_token(on_llm_token)

    print("=" * 60)
    print("  🚐 柴火基地车 Reachy Mini — 皮皮虾")
    print(f"  🎤 {cfg.audio_device or 'Reachy Mini Audio'}")
    print(f"  🤖 {cfg.bailian_llm_model} | 🔊 {cfg.bailian_tts_model}")
    print("=" * 60)
    print("  说 '皮皮虾' 唤醒，Ctrl+C 退出\n")

    try:
        await engine.start()
    except KeyboardInterrupt:
        print("\n\n👋 皮皮虾下线！\n")
    finally:
        _arm_shutdown_backstop()
        await _sleep_reachy_on_shutdown(reachy, cfg)
        await engine.stop()
        await _close_reachy_runtime(reachy)


if __name__ == "__main__":
    main()
