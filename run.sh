#!/usr/bin/env bash
# Run CSTR-RLPD Lab. It is a pure client-side web app; this command only serves
# the static files in frontend/.
#   ./run.sh            -> http://127.0.0.1:8010
#   PORT=9000 ./run.sh
set -e
cd "$(dirname "$0")"
exec python3 serve.py "${PORT:-8010}"
