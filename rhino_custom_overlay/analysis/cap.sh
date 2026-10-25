#!/bin/bash
# usage: ./cap.sh NAME [DEST_FOLDER] ["note"]      (NAME without .npy)
# Same ssh and scp commands as 19-20 Oct. Asks for the board password twice (ssh, then sudo) and once for scp.
# Logs PC clock times (never the board clock) to DEST/captures_log.csv:
#   pc_utc_start         = before ssh (includes your password typing, so it can be 30 s early)
#   pc_utc_after_capture = right after the board script finished = best time stamp (capture ends about 1 to 2 s before it) 
#   pc_utc_after_copy    = after scp
if [ -z "$1" ]; then echo "usage: ./cap.sh NAME [DEST] [\"note\"]"; exit 1; fi
NAME=$1; DEST=${2:-.}; NOTE=${3:-}
HOST=$(hostname -s)
LOG="$DEST/captures_log.csv"
[ -f "$LOG" ] || echo "file,host,pc_utc_start,pc_utc_after_capture,pc_utc_after_copy,note" > "$LOG"
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ); echo "PC time before: $T0"
ssh -t xilinx@192.168.2.99 "sudo -i bash -c 'cd /home/xilinx/pfb_test && python3 s5k16875.py && mv -n capture_k16875.npy $NAME.npy && ls -l $NAME.npy'" || { echo "CAPTURE FAILED, stop"; exit 1; }
T1=$(date -u +%Y-%m-%dT%H:%M:%SZ); echo "PC time after capture: $T1"
scp xilinx@192.168.2.99:/home/xilinx/pfb_test/$NAME.npy "$DEST"/ || { echo "COPY FAILED"; exit 1; }
T2=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "$NAME.npy,$HOST,$T0,$T1,$T2,\"$NOTE\"" >> "$LOG"
ls -l "$DEST/$NAME.npy"; echo "logged to $LOG"
