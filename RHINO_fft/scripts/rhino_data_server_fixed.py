#!/usr/bin/env python

import asyncio
from websockets.asyncio.server import serve
import json
import hashlib
from pickle import loads
import numpy as np
import time

HOSTNAME = "0.0.0.0"   # Accept connections from any machine on the network.
                       # "localhost" would only accept connections from THIS
                       # machine, which blocks the RFSoC from connecting.
SERVER_PORT = 8765

# Directory where received .npy files will be saved
import os
SAVE_DIR = "./received_data"
os.makedirs(SAVE_DIR, exist_ok=True)

async def receive_data_array(websocket):
    
    metadata = ""
    recv_arr = []
    
    # Collect all received data
    async for message in websocket:
        
        # Check received data type
        if isinstance(message, bytes):
            # Data sent as bytes
            recv_arr.append(message)
        else:
            # Metadata sent as JSON
            metadata = json.loads(message)
        
        # Bounce back the message
        #await websocket.send(message, text=False)
    
    # Combine received data
    recv_arr = loads(b"".join(recv_arr))
    print(np.sum(recv_arr))
    
    # Check md5sum
    md5sum = hashlib.md5(recv_arr).hexdigest()
    
    # Verify checksum matches what the client sent (if provided)
    if 'md5sum' in metadata:
        if md5sum == metadata['md5sum']:
            print(f"md5sum OK : {md5sum}")
        else:
            print(f"md5sum MISMATCH!")
            print(f"  Expected : {metadata['md5sum']}")
            print(f"  Got      : {md5sum}")
    else:
        print(f"md5sum : {md5sum}")
    
    print(metadata)
    
    # Save to file
    filename = metadata.get('filename', f"data_{int(time.time())}.npy")
    save_path = os.path.join(SAVE_DIR, filename)
    np.save(save_path, recv_arr)
    print(f"Saved → {save_path}  shape={recv_arr.shape}  sum={np.sum(recv_arr):.4e}")

async def main():
    async with serve(receive_data_array, HOSTNAME, SERVER_PORT) as server:
        await server.serve_forever()

asyncio.run(main())