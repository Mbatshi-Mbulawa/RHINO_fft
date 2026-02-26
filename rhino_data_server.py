#!/usr/bin/env python

import asyncio
from websockets.asyncio.server import serve
import json
import hashlib
from pickle import loads
import numpy as np

HOSTNAME = "localhost"
SERVER_PORT = 8765

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
    print("md5sum:", md5sum)
    print(metadata)
    
    # Save to file
    

async def main():
    async with serve(receive_data_array, HOSTNAME, SERVER_PORT) as server:
        await server.serve_forever()

asyncio.run(main())
