#!/usr/bin/env python

from websockets.sync.client import connect
from pickle import dumps, loads
from sys import getsizeof

import numpy as np
import itertools
import json
import hashlib
import time
import queue
import threading

WEBSOCKETS_MAX_SIZE = int(1e6)
WEBSOCKETS_CHUNK_SIZE = int(1e6)


def send_array(websocket, arr, chunk_size=WEBSOCKETS_CHUNK_SIZE):
    """
    Send an object (e.g. a numpy array) over a websockets connection.
    
    This must be done in chunks if the object is large, as there is a maximum 
    message size for a websockets connection. This function handles the chunking.
    
    Parameters:
        websocket (websockets connection):
            Active websocket connection for sending data.
        arr (array_like):
            Array of data to send over the connection.
        chunk_size (int):
            Size of chunks to break the data up into, in bytes.
    """
    # Send array in chunks
    bytes_arr = dumps(arr)
    
    # Loop over chunks of binary data
    t0 = time.time()
    for chunk in itertools.batched(bytes_arr, chunk_size):
        websocket.send(bytes(chunk), text=False) # send as bytes
    
    print("Send/recv took %6.3f sec" % (time.time() - t0))
    t0 = time.time()
    

def daq_process():
    """
    Mainloop to acquire data and send it to the storage server.
    """
    def worker():
        # Persistent worker process, which collects data from the queue and 
        # sends it to the server
        while True:
            # Get item from queue
            metadata, arr = q.get()
            
            # Open connection and send
            with connect("ws://localhost:8765", max_size=WEBSOCKETS_MAX_SIZE) as websocket:
                
                # Send metadata as (serialised) text
                websocket.send(json.dumps(metadata), text=True)
                
                # Send data array as bytes
                send_array(websocket, arr)
            
            print('Finished item')
            q.task_done()
    
    # Set up queue for data to be sent to the server
    # FIXME: Should set a maxsize to avoid getting backlogged
    q = queue.Queue()
    
    # Fire up the worker thread
    threading.Thread(target=worker, daemon=True).start()
    
    # Loop 5 times to put things in queue
    for i in range(5):
        print("Loop %d" % i)
        arr = (i+1)*np.arange(10000000)
        metadata = {
            'filename': "file%06d.npy" % i, 
            'md5sum':   hashlib.md5(arr).hexdigest()
            }
        print("    Array size in memory: %6.2f MB" % ( getsizeof(arr) / (1024*1024)))
        q.put((metadata, arr))
        # FIXME: Should catch queue.Full exceptions
        #time.sleep(1) # brief sleep
        
    print("Finished adding to queue.")
    q.join()
    print("Finished processing queue.")
    

if __name__ == '__main__':
    # Start the data acquisition process
    daq_process()
