#!/usr/bin/env bash
# Downloads the synthetic telematics data set of So, Boucher and Valdez (2021) into data/ and checks its SHA-256.
# The authors of the data set publish it on http://www2.math.uconn.edu/~valdez/data.html; set SO2021_URL to another
# location of the same file if the default link has moved. Without network access, download the file by hand, put it
# at data/telematics_syn-032021.csv and run this script again to check it.
# Usage: bash data/fetch_so2021.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FILE=telematics_syn-032021.csv
URL="${SO2021_URL:-http://www2.math.uconn.edu/~valdez/$FILE}"
cd "$HERE"
if [ ! -f "$FILE" ]; then
  echo "downloading $URL"
  curl -fL --retry 3 -o "$FILE.part" "$URL"
  mv "$FILE.part" "$FILE"
fi
if sha256sum -c SHA256SUMS; then
  echo "data set ready: data/$FILE"
else
  echo "SHA-256 mismatch: data/$FILE is not the file the results were computed from" >&2
  exit 1
fi
