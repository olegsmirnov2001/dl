#!/usr/bin/env bash
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "must be run as root: sudo $0 <domain>" >&2
  exit 1
fi

DOMAIN="${1:-}"
EMAIL="${2:-}"
if [ -z "$DOMAIN" ] || [[ "$DOMAIN" =~ [[:space:]] ]] || [[ ! "$DOMAIN" =~ ^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
  echo "usage: sudo $0 <domain> [email]" >&2
  echo "       sudo $0 jackalgame.online" >&2
  echo "       sudo $0 jackalgame.online you@example.com" >&2
  exit 1
fi
if [ -n "$EMAIL" ] && [[ ! "$EMAIL" =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]]; then
  echo "invalid email: $EMAIL" >&2
  exit 1
fi

echo
echo ">>> Configuring nginx + Let's Encrypt for: $DOMAIN"
echo

PUBLIC_IP=$(curl -fsS --max-time 5 https://api.ipify.org || curl -fsS --max-time 5 https://ifconfig.me || true)
RESOLVED=$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk '{print $1; exit}' || true)
if [ -n "$PUBLIC_IP" ] && [ -n "$RESOLVED" ] && [ "$PUBLIC_IP" != "$RESOLVED" ]; then
  echo "!! $DOMAIN currently resolves to $RESOLVED"
  echo "!! this server's public IP looks like   $PUBLIC_IP"
  echo "!! Let's Encrypt will fail until the A record points here."
  read -r -p "continue anyway? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "aborting"; exit 1;; esac
elif [ -z "$RESOLVED" ]; then
  echo "!! $DOMAIN does not resolve yet — DNS may still be propagating."
  read -r -p "continue anyway? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "aborting"; exit 1;; esac
else
  echo "DNS OK: $DOMAIN -> $RESOLVED"
fi

echo
echo ">>> installing nginx, certbot, python3-certbot-nginx"
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nginx certbot python3-certbot-nginx

echo
echo ">>> writing /etc/nginx/sites-available/jackal"
cat > /etc/nginx/sites-available/jackal <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    client_max_body_size 1m;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSocket upgrade
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/jackal /etc/nginx/sites-enabled/jackal
rm -f /etc/nginx/sites-enabled/default

echo
echo ">>> validating nginx config"
nginx -t
systemctl enable --now nginx
systemctl reload nginx

if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q '^Status: active'; then
  echo
  echo ">>> opening 80/tcp and 443/tcp in ufw"
  ufw allow 80/tcp || true
  ufw allow 443/tcp || true
fi

echo
echo ">>> requesting Let's Encrypt cert and enabling HTTPS redirect"
CERTBOT_ARGS=(--nginx -d "$DOMAIN" --redirect --agree-tos --non-interactive)
if [ -n "$EMAIL" ]; then
  CERTBOT_ARGS+=(--email "$EMAIL" --no-eff-email)
  echo "    (registering with email: $EMAIL)"
else
  CERTBOT_ARGS+=(--register-unsafely-without-email)
  echo "    (registering without email; you won't get expiry warnings)"
fi
certbot "${CERTBOT_ARGS[@]}"

systemctl enable --now certbot.timer 2>/dev/null || true

echo
echo "all done."
echo "  visit:   https://$DOMAIN/"
echo "  certbot timer status:   systemctl status certbot.timer"
echo
echo "next steps (optional, not done by this script):"
echo "  - lock the uvicorn service to 127.0.0.1 only:"
echo "      edit ~/.config/systemd/user/jackal.service, change --host 0.0.0.0 to --host 127.0.0.1,"
echo "      then 'systemctl --user daemon-reload && systemctl --user restart jackal'."
echo "  - close port 8080 on any cloud firewall (Hetzner/AWS/etc) so traffic only goes through nginx."
