#!/bin/bash
# 加上apt update && apt install -y netcat-openbsd
BASHRC_FILE="${HOME:-/root}/.bashrc"

# host machine ip

IP="127.0.0.1" # for host network mode
# IP="host.docker.internal" # for bridge network mode

if ! grep -q "仅当本地代理端口连通时才设置代理" "$BASHRC_FILE"; then
    cat >> "$BASHRC_FILE" <<EOF
        # 仅当本地代理端口连通时才设置代理
        vpn(){
            export http_proxy=http://$IP:1080
            export https_proxy=http://$IP:1080
        }
        unvpn(){
            unset http_proxy
            unset https_proxy
        }
EOF
fi

exec "$@"

