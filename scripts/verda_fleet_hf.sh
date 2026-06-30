#!/usr/bin/env bash
# verda_fleet_hf.sh — MTEB-PT fleet via HF-RESUME (NO SFS) — runs in ANY location.
# Each instance: pull done results from HF -> only-missing skips them -> a background
# thread syncs new results to HF every ~60s. Frees the fleet from the SFS location-lock
# -> cheapest GPU anywhere. Default PAYG (stable, no preemption).
# IMPORTANT: give parallel runs DISJOINT model lists (each instance has its own local
# cache + only pulls at start, so overlapping lists would double-compute).
# Usage: bash verda_fleet_hf.sh <model1> [<model2> ...]
set -uo pipefail

API="https://api.verda.com/v1"
STATE_DIR="${VERDA_STATE_DIR:-/tmp/verda_fleet_hf_state}"; mkdir -p "$STATE_DIR"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_KEY_ID="50e04745-7ab1-4fa0-978b-bce0bfff60ac"
IMAGE="ubuntu-24.04-cuda-12.8-open-docker"
BRANCH="${MTEB_BRANCH:-portulex-rrip}"
ENV_VERDA="/Users/tardelli/Workplace/embedding-insights/.env"
ENV_HF="/Users/tardelli/Workplace/huggingface-mteb-pt/.env"
HF_REPO="${HF_RESULTS_REPO:-mteb-pt/mteb-pt-results}"
MODELS="$*"
[[ -z "$MODELS" ]] && { echo "usage: bash verda_fleet_hf.sh <model1> [<model2> ...]"; exit 1; }

# Any location (cheap GPUs roam FIN-01/02/03). PAYG price-ascending (stable).
LOCATIONS="${VERDA_LOCATIONS:-FIN-01 FIN-02 FIN-03}"
_DEF="1A6000.10V@False 1RTX6000ADA.10V@False 1A100.40S.22V@False 1A100.22V@False"
INSTANCE_TYPES="${VERDA_INSTANCE_TYPES:-$_DEF}"
IS_SPOT="${VERDA_SPOT:-False}"

_load_env() {
    local f line key val
    for f in "$ENV_VERDA" "$ENV_HF"; do
        [[ -f "$f" ]] || continue
        while IFS= read -r line; do
            line="${line#"${line%%[![:space:]]*}"}"
            [[ -z "$line" || "$line" == "#"* || "$line" != *=* ]] && continue
            key="${line%%=*}"; key="${key%"${key##*[![:space:]]}"}"
            val="${line#*=}"; val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
            [[ -z "${!key:-}" ]] && export "$key=$val"
        done < "$f"
    done
    return 0
}
_token() {
    python3 -c "
import os, urllib.request, json
body=json.dumps({'grant_type':'client_credentials','client_id':os.environ['VERDA_CLIENT_ID'],'client_secret':os.environ['VERDA_CLIENT_SECRET']}).encode()
print(json.loads(urllib.request.urlopen(urllib.request.Request('${API}/oauth2/token',data=body,headers={'Content-Type':'application/json'},method='POST'),timeout=15).read())['access_token'])
"
}
_ssh() {
    local ip; ip=$(cat "$STATE_DIR/instance_ip.txt")
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=40 "root@$ip" "$@"
}

cmd_provision() {
    _load_env; _token > "$STATE_DIR/token.txt"
    local entry t loc spotflag ok=0
    for entry in $INSTANCE_TYPES; do
        t="${entry%@*}"; spotflag="${entry##*@}"; [[ "$spotflag" == "$entry" ]] && spotflag="$IS_SPOT"
      for loc in $LOCATIONS; do
        echo "[hf-fleet] Trying $t @ $loc (spot=$spotflag) ..."
        if python3 -c "
import urllib.request, json, time, sys
API='${API}'; tok=open('${STATE_DIR}/token.txt').read().strip()
H={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/json'}
payload={'instance_type':'${t}','image':'${IMAGE}','hostname':'mtebpt-hf',
         'description':'MTEB-PT hf-resume','location_code':'${loc}',
         'ssh_key_ids':['${SSH_KEY_ID}'],'is_spot':${spotflag},'on_spot_discontinue':'keep_detached'}
import os as _os
_ovs=_os.environ.get('VERDA_OS_SIZE')
if _ovs: payload['os_volume']={'name':'mtebpt-osbig-'+str(int(time.time())),'size':int(_ovs)}
try:
    inst_id=urllib.request.urlopen(urllib.request.Request(API+'/instances',data=json.dumps(payload).encode(),headers=H,method='POST'),timeout=20).read().decode().strip().strip('\"')
except Exception as e:
    print(f'  create failed: {str(e)[:140]}'); sys.exit(2)
open('${STATE_DIR}/instance_id.txt','w').write(inst_id); print(f'  id {inst_id}')
for i in range(180):
    try:
        inst=json.loads(urllib.request.urlopen(urllib.request.Request(f'{API}/instances/{inst_id}',headers=H),timeout=15).read())
        st=inst.get('status','?'); ip=inst.get('ip','')
        print(f'  [{i}] {st} {ip}', flush=True)
        if st in ('running','on') and ip:
            open('${STATE_DIR}/instance_ip.txt','w').write(ip); print('RUNNING'); sys.exit(0)
        if st in ('error','failed','discontinued','deleted'): print('  dead'); sys.exit(3)
    except Exception as e: print(f'  [{i}] {str(e)[:60]}')
    time.sleep(5)
try:
    bd=json.dumps({'action':'delete','id':inst_id,'delete_permanently':True}).encode()
    urllib.request.urlopen(urllib.request.Request(API+'/instances',data=bd,headers=H,method='PUT'),timeout=20); print('  poll timeout -> killed orphan '+inst_id)
except Exception: pass
sys.exit(4)
"; then ok=1; echo "[hf-fleet] Provisioned $t @ $loc at $(cat "$STATE_DIR/instance_ip.txt")"; break 2
        else rm -f "$STATE_DIR/instance_id.txt" "$STATE_DIR/instance_ip.txt"; fi
      done
    done
    [[ "$ok" == 1 ]] || return 1
    sleep 12
}

cmd_setup() {
    local ip; ip=$(cat "$STATE_DIR/instance_ip.txt")
    local i ready=0
    for i in $(seq 1 60); do
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "root@$ip" true 2>/dev/null && { ready=1; break; }
        sleep 10
    done
    [[ "$ready" == 1 ]] || { echo "[hf-fleet] sshd never came up"; return 1; }
    echo "[hf-fleet] installing (no SFS, local cache)..."
    _ssh bash << 'REMOTE'
set -e
for i in $(seq 1 36); do fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock >/dev/null 2>&1 || break; sleep 8; done
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3.12-venv python3-pip >/dev/null 2>&1 || true
mkdir -p /root/mteb_cache
REMOTE
    [[ $? -eq 0 ]] || { echo "[hf-fleet] apt FAILED"; return 1; }
    _ssh "rm -rf /root/mteb-pt /root/venv && git clone -b $BRANCH --depth 1 https://github.com/tardellirs/mteb-pt.git /root/mteb-pt && python3 -m venv --system-site-packages /root/venv && /root/venv/bin/pip install -q -U pip && cd /root/mteb-pt && /root/venv/bin/pip install -q -e . 2>&1 | tail -6 && /root/venv/bin/python -c 'import torch,mteb,mteb_pt.register; assert torch.cuda.is_available(); print(\"[deps] OK cuda=True\")'"
    [[ $? -eq 0 ]] || { echo "[hf-fleet] install FAILED"; return 1; }
    if [[ -n "${MTEB_FLASH_ATTN:-}" ]]; then
        _ssh "MAX_JOBS=4 /root/venv/bin/pip install -q flash-attn --no-build-isolation 2>&1 | tail -3; /root/venv/bin/python -c 'import flash_attn; print(\"[deps] flash-attn\", flash_attn.__version__)' 2>&1 | tail -1"
        echo "[hf-fleet] flash-attn tentado (não-fatal)"
    fi
    if [[ -n "${MTEB_DEVICE_MAP:-}" ]]; then
        _ssh "/root/venv/bin/pip install -q accelerate 2>&1 | tail -1; sed -i 's/_torch.bfloat16}/_torch.bfloat16, \"device_map\": \"${MTEB_DEVICE_MAP}\", \"max_memory\": {0: \"40GiB\", 1: \"40GiB\"}}/g' /root/mteb-pt/scripts/run_mteb_hfresume.py && grep -m1 max_memory /root/mteb-pt/scripts/run_mteb_hfresume.py | cut -c1-90"
        echo "[hf-fleet] device_map=${MTEB_DEVICE_MAP} + max_memory(40/40) + accelerate patched"
    fi
    if [[ -n "${MTEB_LOCAL_RUNNER:-}" ]]; then
        scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /tmp/mteb-pt-clean/scripts/run_mteb_hfresume.py "root@$(cat "$STATE_DIR/instance_ip.txt"):/root/mteb-pt/scripts/run_mteb_hfresume.py" 2>/dev/null && echo "[hf-fleet] runner local (model_prompts) copiado"
    fi
    echo "[hf-fleet] Setup complete (deps OK)"
}

# Launch detached HF-resume eval (pull from HF -> only-missing -> sync to HF) + poll.
cmd_run() {
    _load_env; local hf_tok="${HF_TOKEN:-}"
    echo "[hf-fleet] === Running [$MODELS] (HF-resume, repo=$HF_REPO) ==="
    _ssh "cd /root/mteb-pt && HF_TOKEN='$hf_tok' HF_RESULTS_REPO='$HF_REPO' HF_HUB_CACHE=/root/hfmodels HF_DATASETS_CACHE=/root/hfdata MTEB_CACHE=/root/mteb_cache HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1} HF_SYNC_SECONDS=${HF_SYNC_SECONDS:-60} MTEB_BATCH_SIZE=${MTEB_BATCH_SIZE:-256} MTEB_TASKS='${MTEB_TASKS:-}' MTEB_MODEL_PROMPTS_B64='${MTEB_MODEL_PROMPTS_B64:-}' MTEB_OVERWRITE='${MTEB_OVERWRITE:-only-missing}' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup /root/venv/bin/python scripts/run_mteb_hfresume.py $MODELS > /root/run.log 2>&1 < /dev/null & echo '[remote] launched'"
    local i state last
    for i in $(seq 1 240); do
        sleep 30
        state=$(_ssh "if grep -q 'FLEET RUN COMPLETE' /root/run.log 2>/dev/null; then echo DONE; elif pgrep -f run_mteb_hfresume >/dev/null; then echo RUN; else echo DEAD; fi" 2>/dev/null || echo SSHERR)
        last=$(_ssh "tail -1 /root/run.log 2>/dev/null | tr -d '\r' | tail -c 80" 2>/dev/null)
        echo "[hf-fleet] [$i] $state | $last"
        [[ "$state" == "DONE" ]] && { echo "DONE" > "$STATE_DIR/run_state"; return 0; }
        [[ "$state" == "DEAD" || "$state" == "SSHERR" ]] && { echo "DEAD" > "$STATE_DIR/run_state"; return 0; }
    done
    echo "TIMEOUT" > "$STATE_DIR/run_state"
}

_terminate_instance() {
    _token > "$STATE_DIR/token.txt" 2>/dev/null || true
    local iid; iid=$(cat "$STATE_DIR/instance_id.txt" 2>/dev/null || echo "")
    [[ -n "$iid" ]] && python3 -c "
import urllib.request, json
tok=open('${STATE_DIR}/token.txt').read().strip()
H={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/json'}
b=json.dumps({'action':'delete','id':'${iid}','delete_permanently':True}).encode()
try: urllib.request.urlopen(urllib.request.Request('${API}/instances',data=b,headers=H,method='PUT'),timeout=20)
except Exception: pass
import time; time.sleep(3)
vols=json.loads(urllib.request.urlopen(urllib.request.Request('${API}/volumes',headers=H),timeout=15).read())
for v in vols:
    if v.get('status')=='detached' and str(v.get('name','')).startswith('OS-'):
        try: urllib.request.urlopen(urllib.request.Request('${API}/volumes/'+v['id']+'?is_permanent=true',headers=H,method='DELETE'),timeout=20); print('  cleaned OS disk',v['id'])
        except Exception: pass
"
    rm -f "$STATE_DIR/instance_id.txt" "$STATE_DIR/instance_ip.txt"
}

# ── main: deadline loop. Results live on HF (continuous sync), so DONE just tears down. ──
trap '_terminate_instance >> "$STATE_DIR/teardown.log" 2>&1 || true' EXIT INT TERM
_load_env
done=0; _start=$(date +%s); _deadline=$((_start + ${MTEB_MAX_HOURS:-8} * 3600)); attempt=0
while [[ $(date +%s) -lt $_deadline ]]; do
    attempt=$((attempt + 1))
    echo "===== [hf-fleet] attempt $attempt (elapsed $(( ($(date +%s) - _start) / 60 ))min) ====="
    cmd_provision || { echo "[hf-fleet] no GPU available — wait 60s + retry"; sleep 60; continue; }
    cmd_setup || { echo "[hf-fleet] setup failed — terminate + retry"; _terminate_instance; sleep 10; continue; }
    cmd_run
    if [[ "$(cat "$STATE_DIR/run_state" 2>/dev/null)" == "DONE" ]]; then
        echo "[hf-fleet] === ALL MODELS DONE — run.log model/FAILED lines: ==="
        _ssh "grep -aE 'model:|FAILED|done in|Error|Traceback' /root/run.log | tail -25" 2>/dev/null
        _bip=$(cat "$STATE_DIR/instance_ip.txt" 2>/dev/null); mkdir -p "/tmp/verda_backup/$_bip"
        rsync -rz --ignore-existing --include='*/' --include='*.json' --exclude='*' -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15" "root@$_bip:/root/mteb_cache/results/" "/tmp/verda_backup/$_bip/" 2>/dev/null && echo "[hf-fleet] backup final -> /tmp/verda_backup/$_bip ANTES do teardown"
        done=1; _terminate_instance; break
    fi
    echo "[hf-fleet] instance died — retry (resume via HF only-missing)"
    _terminate_instance; sleep 8
done
trap - EXIT
[[ "$done" == 1 ]] && echo "[hf-fleet] COMPLETE" || echo "[hf-fleet] deadline reached (results on HF)"
