#!/usr/bin/env bash
# Manage the Chaihuo Reachy dashboard and its project-owned SDK daemon.

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${CHAIHUO_SERVICE_STATE_DIR:-${PROJECT_DIR}/state}"
PID_FILE="${STATE_DIR}/service.pid"
LOG_FILE="${CHAIHUO_SERVICE_LOG_FILE:-${STATE_DIR}/service.log}"
PYTHON="${CHAIHUO_SERVICE_PYTHON:-${PROJECT_DIR}/.venv/bin/python}"
START_TIMEOUT_S="${CHAIHUO_SERVICE_START_TIMEOUT_S:-45}"
STOP_TIMEOUT_S="${CHAIHUO_SERVICE_STOP_TIMEOUT_S:-35}"

read_dotenv_value() {
    local name="$1"
    local file="${PROJECT_DIR}/.env"
    [[ -f "${file}" ]] || return 0
    awk -F= -v key="${name}" '
        $0 !~ /^[[:space:]]*#/ && $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
            value = substr($0, index($0, "=") + 1)
            gsub(/^[[:space:]\"'\'' ]+|[[:space:]\"'\''\r ]+$/, "", value)
            print value
            exit
        }
    ' "${file}"
}

DASHBOARD_PORT="${REACHY_DASHBOARD_PORT:-$(read_dotenv_value REACHY_DASHBOARD_PORT)}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8640}"
DAEMON_STATE_FILE="${REACHY_DAEMON_STATE_FILE:-$(read_dotenv_value REACHY_DAEMON_STATE_FILE)}"
DAEMON_STATE_FILE="${DAEMON_STATE_FILE:-state/daemon.json}"
if [[ "${DAEMON_STATE_FILE}" != /* ]]; then
    DAEMON_STATE_FILE="${PROJECT_DIR}/${DAEMON_STATE_FILE}"
fi
HEALTH_URL="${CHAIHUO_SERVICE_HEALTH_URL:-http://127.0.0.1:${DASHBOARD_PORT}/healthz}"

usage() {
    printf '%s\n' \
        "用法: ./reachy-service.sh {start|stop|restart|status|logs} [启动参数]" \
        "" \
        "  start [参数]  后台启动服务，例如: start --target jetson" \
        "  stop          安全休眠机器人并关闭服务和本项目 daemon" \
        "  restart       安全关闭后重新启动" \
        "  status        查看进程、Dashboard 和 daemon 状态" \
        "  logs          持续查看运行日志"
}

read_pid() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(tr -dc '0-9' < "${PID_FILE}")"
    [[ -n "${pid}" ]] || return 1
    printf '%s' "${pid}"
}

process_command() {
    ps -p "$1" -o command= 2>/dev/null || true
}

is_owned_service() {
    local pid="$1"
    kill -0 "${pid}" 2>/dev/null || return 1
    local command
    command="$(process_command "${pid}")"
    [[ "${command}" == *"chaihuo_reachy.main"* ]] || return 1
    [[ "${command}" == *"dashboard"* ]] || return 1
    if [[ -e "/proc/${pid}/cwd" ]]; then
        [[ "$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)" == "${PROJECT_DIR}" ]] || return 1
    fi
    return 0
}

health_ok() {
    if command -v curl >/dev/null 2>&1; then
        curl --fail --silent --show-error --max-time 1 "${HEALTH_URL}" >/dev/null 2>&1
        return $?
    fi
    "${PYTHON}" - "${HEALTH_URL}" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=1) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
}

cleanup_stale_pid() {
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && ! is_owned_service "${pid}"; then
        rm -f "${PID_FILE}"
    fi
}

recover_owned_daemon() {
    [[ -x "${PYTHON}" ]] || return 0
    (
        cd "${PROJECT_DIR}" || exit 1
        "${PYTHON}" -c \
            'import sys; from chaihuo_reachy import daemon_runtime; raise SystemExit(0 if daemon_runtime.terminate_owned_state(sys.argv[1]) else 1)' \
            "${DAEMON_STATE_FILE}"
    ) >/dev/null 2>&1 || true
}

start_service() {
    mkdir -p "${STATE_DIR}"
    cleanup_stale_pid
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && is_owned_service "${pid}"; then
        printf '✅ 服务已经运行（PID %s）: http://localhost:%s\n' "${pid}" "${DASHBOARD_PORT}"
        return 0
    fi
    if [[ ! -x "${PYTHON}" ]]; then
        printf '❌ 找不到虚拟环境 Python: %s\n请先运行: uv sync\n' "${PYTHON}" >&2
        return 1
    fi

    printf '🚀 正在启动皮皮虾服务...\n'
    printf '\n[%s] service start\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    (
        cd "${PROJECT_DIR}" || exit 1
        exec nohup env PYTHONUNBUFFERED=1 "${PYTHON}" -m chaihuo_reachy.main dashboard "$@"
    ) >> "${LOG_FILE}" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"

    # Give nohup/env a brief moment to exec Python before validating the
    # command line. This avoids treating a healthy launch as a stale PID.
    local launch_checks=0
    while ! is_owned_service "${pid}" && kill -0 "${pid}" 2>/dev/null && (( launch_checks < 30 )); do
        sleep 0.1
        launch_checks=$((launch_checks + 1))
    done

    local waited=0
    while (( waited < START_TIMEOUT_S )); do
        if ! is_owned_service "${pid}"; then
            printf '❌ 服务启动失败，最近日志：\n' >&2
            tail -n 30 "${LOG_FILE}" >&2 || true
            rm -f "${PID_FILE}"
            recover_owned_daemon
            return 1
        fi
        if health_ok; then
            printf '✅ 服务启动成功（PID %s）\n🌐 http://localhost:%s\n📝 日志: %s\n' \
                "${pid}" "${DASHBOARD_PORT}" "${LOG_FILE}"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    printf '⚠️ 服务进程仍在运行，但 %s 秒内未通过健康检查。\n' "${START_TIMEOUT_S}" >&2
    printf '请运行 ./reachy-service.sh logs 查看初始化状态。\n' >&2
    return 1
}

stop_service() {
    cleanup_stale_pid
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    if [[ -z "${pid}" ]]; then
        recover_owned_daemon
        printf 'ℹ️ 服务没有运行；已检查并清理本项目拥有的残留 daemon。\n'
        return 0
    fi
    if ! is_owned_service "${pid}"; then
        printf '⚠️ PID 文件不属于当前服务，不会终止该进程。\n' >&2
        rm -f "${PID_FILE}"
        recover_owned_daemon
        return 1
    fi

    printf '🛌 正在安全关闭服务（PID %s），机器人将进入休眠...\n' "${pid}"
    kill -INT "${pid}" 2>/dev/null || true
    local waited=0
    while (( waited < STOP_TIMEOUT_S )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "${pid}" 2>/dev/null; then
        printf '⚠️ 优雅关闭超时，发送 SIGTERM...\n' >&2
        kill -TERM "${pid}" 2>/dev/null || true
        sleep 3
    fi
    if kill -0 "${pid}" 2>/dev/null; then
        printf '⚠️ 进程仍未退出，强制释放服务资源。\n' >&2
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
    recover_owned_daemon
    if health_ok; then
        printf '⚠️ Dashboard 端口仍有服务响应；该监听者不属于本脚本，因此没有误杀。\n' >&2
        return 1
    fi
    printf '✅ 服务已关闭，Dashboard、音频设备和本项目 daemon 已释放。\n'
}

show_status() {
    cleanup_stale_pid
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && is_owned_service "${pid}"; then
        printf '✅ 服务运行中（PID %s）\n' "${pid}"
        if health_ok; then
            printf '✅ Dashboard 健康: http://localhost:%s\n' "${DASHBOARD_PORT}"
        else
            printf '⚠️ Dashboard 尚未就绪，请查看日志。\n'
        fi
        printf '📝 日志: %s\n' "${LOG_FILE}"
        return 0
    fi
    printf '⏹️ 服务未运行\n'
    return 1
}

command="${1:-}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "${command}" in
    start)
        start_service "$@"
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service "$@"
        ;;
    status)
        show_status
        ;;
    logs)
        mkdir -p "${STATE_DIR}"
        touch "${LOG_FILE}"
        tail -n 100 -f "${LOG_FILE}"
        ;;
    *)
        usage
        exit 2
        ;;
esac
