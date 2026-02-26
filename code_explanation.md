# Line-by-Line Code Explanation
## Spectrum_acquisition_demo.py

---

## SECTION 1: The Shebang Line and Imports (Lines 1-58)

---

### Line 1
```python
#!/usr/bin/env python3
```
This is called a **shebang line**. It tells the operating system: 
"When you run this file directly, use Python 3 to execute it."

Without this, you'd always have to type:
```bash
python3 rfsoc_rhino_simplified.py
```
With it, you can also just type:
```bash
./rfsoc_rhino_simplified.py
```
The `/usr/bin/env python3` part means "find Python 3 wherever it's installed" 
rather than assuming it's at a fixed location.

---

### Lines 2-48: The Docstring
```python
"""
RHINO 21cm Global Signal Acquisition...
"""
```
Triple quotes create a **module docstring** — a description of the entire file.
This is not code that runs; it's documentation.
It also serves a second purpose: if someone types `help(this_module)` in Python,
this text is what they'll see.

---

### Lines 50-58: Imports
```python
import numpy as np
import time
import sys
import hashlib
import json
from datetime import datetime
from websockets.sync.client import connect
from pickle import dumps
import itertools
```

These lines load external libraries into memory so we can use them.
Think of it like loading tools into your workshop before you start building.

**Each import explained:**

```python
import numpy as np
```
NumPy is the core scientific computing library in Python.
It provides the `array` data type, which is how we store spectra.
`as np` means we can write `np.zeros(...)` instead of `numpy.zeros(...)`.
Without NumPy, Python has no efficient way to handle large arrays of numbers.

```python
import time
```
Python's built-in timing library.
We use it for two things:
- `time.time()` returns the current time as a number (seconds since 1970)
- Subtracting two `time.time()` calls gives you elapsed time

```python
import sys
```
Python's system library.
We specifically use `sys.exit(1)` to deliberately stop the program
when something goes wrong. The number `1` means "exited with error"
(by convention, `0` means success, anything else means failure).

```python
import hashlib
```
Provides cryptographic hash functions, specifically MD5.
We use this to generate a "fingerprint" of our data.
If we send data over the network and the fingerprint on arrival
matches the fingerprint we computed before sending, the data
arrived intact and wasn't corrupted.

```python
import json
```
JSON (JavaScript Object Notation) is a standard format for
structured text data. Looks like:
```json
{"key": "value", "number": 42}
```
We use it to send metadata (timestamps, settings, etc.) over
the network in a format both the RFSoC and logging computer understand.

```python
from datetime import datetime
```
Provides the `datetime` class for working with dates and times.
We use `datetime.utcnow().isoformat()` to generate timestamps
like `"2026-02-11T14:30:00.123456"` for our filenames and metadata.

```python
from websockets.sync.client import connect
```
This imports the WebSocket client from the `websockets` library.
A WebSocket is a type of network connection that stays open
and allows both sides to send messages back and forth.
We use this to send data from the RFSoC to the logging computer.
`sync` means synchronous (the code waits for each send to finish
before moving on), as opposed to `async` which would run in the background.

```python
from pickle import dumps
```
`pickle` is Python's serialization library.
Serialization = converting a Python object (like a numpy array)
into raw bytes that can be saved to disk or sent over a network.
`dumps` = "dump to string (bytes)" — specifically converts
a Python object into a bytes object.
`loads` (not imported here, used in the server) does the reverse.

```python
import itertools
```
A Python standard library for working with iterators efficiently.
We specifically use `itertools.batched()` which splits a sequence
into fixed-size chunks — essential for chunked network transmission.

---

## SECTION 2: The Configuration Class (Lines 64-147)

---

```python
class RHINOConfig:
```
A **class** in Python is a container for related data and functions.
Here `RHINOConfig` is used purely as a container for settings.
Think of it as a settings panel at the top of the script.
All values here are **class attributes** — they belong to the class itself,
not to any particular instance. You access them as `RHINOConfig.FFT_SIZE` etc.

---

### Network Settings (Lines 78-81)

```python
SERVER_HOSTNAME = "192.168.1.100"
```
The IP address of the computer running the data logging server.
An IP address is like a street address on a network — it uniquely
identifies one computer. You must change this to match your actual
logging computer's IP address before running.

```python
SERVER_PORT = 8765
```
A port number is like a door number on a building (the IP address is the building).
Port 8765 is arbitrary — it just needs to match the port in `rhino_data_server.py`.
Both client and server must agree on the same port.
Valid ports: 0-65535. Ports below 1024 are reserved for system services.

```python
WEBSOCKETS_MAX_SIZE = int(1e6)
```
`int(1e6)` = `int(1,000,000)` = `1000000` bytes = 1 MB.
This is the maximum size of a single WebSocket message.
Larger data must be broken into chunks (see `send_array`).
`1e6` is scientific notation: 1 × 10^6.

```python
WEBSOCKETS_CHUNK_SIZE = int(1e6)
```
When sending data larger than 1 MB, we break it into chunks of this size.
Setting both MAX_SIZE and CHUNK_SIZE to 1 MB means each chunk
is exactly the maximum allowed message size.

---

### RF / Sampling Settings (Lines 91-92)

```python
TARGET_SAMPLE_RATE_MHZ = 300.0
```
We want the ADC to sample at 300 million times per second (300 MHz).
Why 300 MHz?
- Nyquist theorem: sample rate ≥ 2 × highest frequency of interest
- FM radio band ends at ~108 MHz
- 2 × 108 = 216 MHz minimum
- 300 MHz gives comfortable headroom and covers FM without aliasing

```python
DECIMATION_FACTOR = 1
```
Decimation means reducing the sample rate by discarding samples.
For example, decimation by 8 keeps every 8th sample,
reducing the effective sample rate by 8×.
We set this to 1 (no decimation) because:
- We want the full 300 MHz bandwidth
- We're doing direct sampling, not DDC-based narrowband reception
- Reducing sample rate would reduce our frequency coverage

---

### FFT Settings (Lines 97-108)

```python
FFT_SIZE = 16384
```
This is the number of samples fed into one FFT computation.
It directly determines frequency resolution:

```
Frequency Resolution = Sample_Rate / FFT_Size
                     = 300 MHz / 16384
                     = 18.3 kHz per frequency bin
```

So the output of the FFT is an array of 16384 numbers,
where each number represents the power in a 18.3 kHz wide frequency channel.

Why a power of 2 (16384 = 2^14)?
The FFT algorithm (Cooley-Tukey) is most efficient when N is a power of 2.

```python
NUMBER_OF_FRAMES = 32
```
This controls hardware-level averaging inside the FPGA.
The FPGA FFT IP core computes 32 consecutive FFTs and averages them
before returning the result to Python.
This reduces noise (signal averages coherently, noise averages incoherently).

```python
WINDOW_TYPE = 'blackman'
```
Before computing the FFT, each block of samples is multiplied
by a window function. Without windowing, the sharp edges at the start
and end of each FFT block cause "spectral leakage" — power from
one frequency bin bleeding into neighboring bins.
The Hanning window smoothly tapers the signal to zero at both edges,
dramatically reducing this leakage.
Other options: 'rectangular' (no window, worst leakage),
'hamming', 'blackman' (similar to Hanning, slightly different trade-offs).

```python
SPECTRUM_TYPE = 'power'
```
This tells the hardware what to output after the FFT:
- `'magnitude'` = |FFT| (square root of power)
- `'power'` = |FFT|² (what the PhD student asked for)
- `'log'` = 10*log10(|FFT|²) = dB scale
Power is standard in radio astronomy because it's directly
proportional to the signal intensity we care about measuring.

---

### Time Averaging Settings (Lines 117-123)

```python
INTEGRATION_TIME_SECONDS = 1.0
```
After the hardware computes spectra, the Python software averages them
over this time period. The PhD student said "like 1 s".
This is called the **integration time**.

Longer integration = lower noise (noise reduces as √N where N = number averaged)
Shorter integration = better time resolution (can see rapid changes)
For the 21cm signal, which is constant over human timescales, longer is better.

---

### Data Collection Settings (Lines 128-130)

```python
ACQUISITION_DURATION = 3600
```
Total time to run: 3600 seconds = 1 hour.
After this time, the acquisition loop stops cleanly.

```python
SPECTRA_PER_FILE = 60
```
How many time-averaged spectra to bundle into each transmitted file.
With 1-second integration, 60 spectra = 60 seconds = 1 minute per file.
Smaller files transmit faster but create more overhead.
Larger files are more efficient but you lose data if transmission fails.

---

### Other Settings (Lines 135-147)

```python
CALIBRATION_MODE = 1
```
The RFSoC ADC has different calibration modes for the analog front-end.
Mode 1 is standard. The rfsoc_sam library accepts 1 or 2.

```python
ENABLE_MD5_CHECKSUM = True
```
If True, we compute an MD5 hash of the data before sending it.
The server can verify the hash on arrival to confirm data integrity.
MD5 is a function that takes any input and produces a fixed 32-character
"fingerprint". Even one bit of corruption changes the fingerprint completely.

```python
VERBOSE = True
SAVE_LOCAL_COPY = False
LOCAL_DATA_DIR = "./data"
```
`VERBOSE = True` means print detailed progress messages.
`SAVE_LOCAL_COPY = False` means don't also save data on the RFSoC board.
`LOCAL_DATA_DIR = "./data"` is the directory to use IF local saving is enabled.
`./` means "current directory".

---

## SECTION 3: send_array() Function (Lines 154-182)

---

```python
def send_array(websocket, arr, chunk_size=RHINOConfig.WEBSOCKETS_CHUNK_SIZE):
```
This **function** takes three inputs:
- `websocket`: An open network connection (like a phone call that's already connected)
- `arr`: The numpy array to send
- `chunk_size`: How large each piece should be (defaults to 1 MB)

The `=RHINOConfig.WEBSOCKETS_CHUNK_SIZE` part means chunk_size has a
**default value** — you don't have to provide it, it uses 1 MB automatically.

```python
bytes_arr = dumps(arr)
```
`dumps` (from pickle) converts the numpy array into raw bytes.
Why? Networks send bytes, not Python objects.
The result `bytes_arr` is the array serialized into a long sequence of bytes
that can be transmitted and later reconstructed with `loads()` (on the server).

Think of it like this: if the array is a book,
`dumps` puts the book through a shredder into numbered pieces,
and `loads` reassembles the pieces back into a book.

```python
if RHINOConfig.VERBOSE:
    print(f"    Serialized array size: {len(bytes_arr) / (1024*1024):.2f} MB")
```
`if RHINOConfig.VERBOSE:` — only print if we're in verbose mode.
`len(bytes_arr)` counts the number of bytes.
`/ (1024*1024)` converts bytes → megabytes (1 MB = 1024 × 1024 bytes).
`:.2f` in the f-string formats the number to 2 decimal places.

```python
t_start = time.time()
chunk_count = 0
```
`time.time()` returns the current time as a floating point number
(seconds since 1 January 1970 — the Unix epoch).
We save this so we can calculate how long transmission took.
`chunk_count` tracks how many chunks we've sent (starts at 0).

```python
for chunk in itertools.batched(bytes_arr, chunk_size):
    websocket.send(bytes(chunk), text=False)
    chunk_count += 1
```
`itertools.batched(bytes_arr, chunk_size)` splits `bytes_arr` into
pieces of `chunk_size` bytes each (1 MB chunks).

The `for` loop iterates over each chunk. For each one:
- `websocket.send(bytes(chunk), text=False)`: Send this chunk over the network
  - `bytes(chunk)`: Ensure it's a bytes object
  - `text=False`: Send as binary data (not text/JSON)
- `chunk_count += 1`: Increment the counter

```python
transmission_time = time.time() - t_start
```
`time.time()` again — now the current time.
Subtracting the start time gives us how many seconds transmission took.

```python
if RHINOConfig.VERBOSE:
    print(f"    Transmitted {chunk_count} chunks in {transmission_time:.3f} seconds")
    print(f"    Throughput: {(len(bytes_arr)/(1024*1024))/transmission_time:.2f} MB/s")
```
Reports:
- How many chunks were sent
- How long it took (3 decimal places = millisecond precision)
- Network throughput in MB/s (data size ÷ time)

---

## SECTION 4: transmit_spectrum_data() Function (Lines 185-222)

---

```python
def transmit_spectrum_data(spectrum_data, metadata):
```
This function handles the entire transmission of one batch:
- `spectrum_data`: 2D numpy array of spectra (shape: [60, 16384])
- `metadata`: Python dictionary of information about the data

```python
server_url = f"ws://{RHINOConfig.SERVER_HOSTNAME}:{RHINOConfig.SERVER_PORT}"
```
`f"..."` is an **f-string** — Python's way of embedding variables in strings.
`{RHINOConfig.SERVER_HOSTNAME}` inserts the IP address.
`ws://` is the WebSocket protocol prefix (like `http://` but for WebSockets).
Result example: `"ws://192.168.1.100:8765"`

```python
try:
    ...
except ConnectionRefusedError:
    ...
except Exception as e:
    ...
```
This is a **try/except block** — Python's error handling mechanism.
- `try:` — attempt the code inside
- `except ConnectionRefusedError:` — if the server isn't running, catch this specific error
- `except Exception as e:` — catch any other error, store it in variable `e`

Without try/except, if the server isn't running, the whole program would crash.
With it, we print a helpful message and re-raise the error for the caller to handle.

```python
with connect(server_url, max_size=RHINOConfig.WEBSOCKETS_MAX_SIZE) as websocket:
```
`with ... as websocket:` is a **context manager**.
It automatically:
1. Opens the WebSocket connection when entering the block
2. Closes the connection when leaving the block (even if an error occurs)

`connect(server_url, max_size=...)` establishes the connection.
`max_size` sets the maximum message size the connection will accept.

```python
websocket.send(json.dumps(metadata), text=True)
```
`json.dumps(metadata)` converts the Python dictionary to a JSON string.
Example output: `'{"filename": "rhino_...", "fft_size": 16384, ...}'`

`websocket.send(..., text=True)` sends this as a text message.
The server knows this is metadata because it's text (not binary).

```python
send_array(websocket, spectrum_data)
```
Calls the `send_array()` function we just explained above.
Sends the actual spectrum data as binary in chunks.

```python
print(f"[TRANSMISSION] Successfully transmitted: {metadata['filename']}")
```
`metadata['filename']` accesses the 'filename' key from the dictionary.
This is only reached if no exception was raised above.

---

## SECTION 5: The RHINOAcquisition Class (Lines 229-570)

---

```python
class RHINOAcquisition:
```
This is the main class. Unlike `RHINOConfig` (which just stores settings),
`RHINOAcquisition` has both data (attributes) AND behaviour (methods/functions).
When you write `acquisition = RHINOAcquisition()`, Python creates an
**instance** of this class — a specific object with its own data.

---

### __init__() Method (Lines 240-283)

```python
def __init__(self, overlay_path=None):
```
`__init__` is a special Python method called a **constructor**.
It runs automatically when you create a new instance: `RHINOAcquisition()`.

`self` refers to the specific instance being created.
When you write `self.receiver = ...`, you're storing data ON that instance.
`overlay_path=None` means: if you don't provide a path, use None (use default).

```python
print("="*70)
```
`"="*70` repeats the string "=" 70 times.
Result: `"======================================================================"`
This creates a visual separator line in the console output.

```python
try:
    import pynq
    from rfsoc_sam import overlay
    self.pynq_available = True
except ImportError as e:
    ...
    self.pynq_available = False
    return
```
Why import inside a function?

These imports are done HERE rather than at the top of the file because:
- If they're at the top, the whole script fails to start if PYNQ isn't installed
- If they're here, the script can at least start and give a clear error message

`ImportError` is raised when Python can't find a library.
`self.pynq_available = False` marks that PYNQ isn't available.
`return` exits `__init__` early — the rest of setup doesn't run.

```python
if overlay_path is None:
    self.ol = overlay.SpectrumAnalyserOverlay()
else:
    self.ol = pynq.Overlay(overlay_path)
```
If no path was given, use the default rfsoc_sam overlay.
If a path was given, load that specific bitstream.

A PYNQ overlay is the FPGA design loaded onto the chip.
It contains the FFT IP core, DMA, RF data converter settings etc.
Loading an overlay programs the FPGA with that design.

`self.ol` stores the overlay object — we'll use it to access hardware components.

```python
self.receiver = self.ol.radio.receiver
```
`self.ol` is the overlay (the entire FPGA design).
`.radio` accesses the radio subsystem within the overlay.
`.receiver` accesses the receiver component.

This follows the rfsoc_sam library structure where hardware is
organized as: `overlay → radio → receiver/transmitter`.

```python
self._configure_system()
```
Calls the configuration method (explained next).
The underscore prefix `_` is a Python convention meaning
"this method is internal — external code shouldn't call it directly".

```python
self.total_spectra_collected = 0
self.total_files_transmitted = 0
self.acquisition_start_time = None
```
Initialize counters to track how much data we've collected.
`None` is Python's way of saying "no value yet" — we'll set this
when acquisition actually starts.

---

### _configure_system() Method (Lines 286-361)

```python
self.receiver.analyser.decimation_factor = RHINOConfig.DECIMATION_FACTOR
```
This writes to a hardware register on the RFSoC.
Under the hood, PYNQ uses AXI-Lite (a memory-mapped bus protocol)
to write the value `1` to the decimation control register in the FPGA.
The FPGA's decimation block reads this register and sets itself to pass
all samples through unmodified (decimation by 1 = no decimation).

```python
self.adc_sample_rate = self.receiver.analyser.sample_frequency
```
Reads the actual ADC sample rate from the hardware.
The FPGA knows its own clock rate, and this property reads it.
We store it in `self.adc_sample_rate` for calculations throughout the class.

```python
target_rate = RHINOConfig.TARGET_SAMPLE_RATE_MHZ * 1e6
rate_error = abs(self.adc_sample_rate - target_rate) / target_rate
```
Convert target rate from MHz to Hz (×1e6 = ×1,000,000).
Calculate the fractional error between actual and target rates.
`abs()` gives absolute value (always positive).
If actual = 300 MHz and target = 300 MHz: error = 0.
If actual = 295 MHz and target = 300 MHz: error = 0.0167 = 1.7%.

```python
if rate_error > 0.05:
```
`0.05` = 5%. If the actual rate differs from target by more than 5%,
print a warning. This is a sanity check — if the hardware isn't
configured as expected, you want to know before collecting hours of data.

```python
nyquist_freq = (self.adc_sample_rate / 2) / 1e6
```
Nyquist theorem: maximum frequency you can represent = sample_rate / 2.
`/ 1e6` converts Hz to MHz for display.
With 300 MHz sampling: Nyquist = 150 MHz.

```python
self.receiver.analyser.fftsize = RHINOConfig.FFT_SIZE
freq_resolution = (self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3
```
Sets the FFT size in the hardware IP core.
Calculates frequency resolution in kHz (÷1e3 converts Hz to kHz).
At 300 MHz with 16384 points: 300e6 / 16384 / 1e3 = 18.3 kHz.

```python
self.receiver.analyser.number_frames = RHINOConfig.NUMBER_OF_FRAMES
```
Tells the FPGA to average 32 consecutive FFTs before returning a result.

```python
self.spectrum_time = (RHINOConfig.NUMBER_OF_FRAMES * 
                     RHINOConfig.FFT_SIZE / self.adc_sample_rate)
```
Calculates how long one hardware-averaged spectrum takes:

```
spectrum_time = Frames × FFT_size / sample_rate
              = 32 × 16384 / 300,000,000
              = 524,288 / 300,000,000
              = 0.00175 seconds
              = 1.75 milliseconds
```

In words: 16384 samples × 32 frames = 524,288 samples total.
At 300 million samples per second, that takes 1.75 ms.

```python
self.spectra_per_integration = max(1, int(
    RHINOConfig.INTEGRATION_TIME_SECONDS / self.spectrum_time
))
```
How many hardware spectra should be averaged for 1 second?

```
spectra_per_integration = 1.0 sec / 0.00175 sec
                        = 571 spectra
```

`int(...)` rounds down to an integer (can't capture 571.4 spectra).
`max(1, ...)` ensures we always capture at least 1 spectrum,
even if the integration time is shorter than one spectrum duration.

```python
self.frequency_axis = np.arange(RHINOConfig.FFT_SIZE) * (
    self.adc_sample_rate / RHINOConfig.FFT_SIZE
) / 1e6
```
Creates an array of frequency values corresponding to each FFT bin.

`np.arange(16384)` creates: `[0, 1, 2, 3, ..., 16383]`

Multiplying by `(sample_rate / FFT_size)` converts bin numbers to Hz:
`[0, 18311, 36621, ..., 300,000,000 - 18311]` Hz

Dividing by `1e6` converts to MHz:
`[0.0, 0.0183, 0.0366, ..., 149.98]` MHz

Result: `self.frequency_axis[i]` gives the frequency of FFT bin `i` in MHz.

---

### capture_single_spectrum() Method (Lines 364-377)

```python
def capture_single_spectrum(self):
    spectrum = self.receiver.analyser.spectrum_data()
    return spectrum
```
This is intentionally simple — just one line of actual code.

`self.receiver.analyser.spectrum_data()` triggers the complete hardware pipeline:
1. Tells the DMA to start capturing from the FFT output
2. Waits for 32 FFT frames to be computed and averaged (1.75 ms)
3. DMA transfers the result from FPGA memory to CPU-accessible RAM
4. Returns the data as a numpy array of shape `(16384,)`

This function is separated from the averaging function so that
the two levels of averaging are clearly distinct:
- **Hardware averaging**: 32 frames inside the FPGA (this function)
- **Software time averaging**: 571 calls over 1 second (next function)

---

### capture_time_averaged_spectrum() Method (Lines 380-405)

```python
accumulated = np.zeros(RHINOConfig.FFT_SIZE, dtype=np.float64)
```
Creates an array of 16384 zeros.
`dtype=np.float64` = 64-bit floating point (double precision).
We use float64 here (not float32) because we're accumulating 571 spectra —
repeatedly adding float32 numbers causes precision loss through rounding.
Higher precision during accumulation, then convert back to float32 at the end.

```python
for i in range(self.spectra_per_integration):
    spectrum = self.capture_single_spectrum()
    accumulated += spectrum
    self.total_spectra_collected += 1
```
`range(571)` generates numbers 0 to 570.
The loop runs 571 times. Each iteration:
1. Captures one hardware-averaged spectrum (1.75 ms wait)
2. `accumulated += spectrum` adds the new spectrum to the running total
   (this is element-wise addition — adds each bin to the corresponding bin)
3. Increments the counter (for statistics)

After the loop, `accumulated` contains the SUM of 571 spectra.

```python
averaged_spectrum = accumulated / self.spectra_per_integration
```
Divides by 571 to get the MEAN (sum ÷ count = average).
This is the time-averaged power spectrum.

```python
return averaged_spectrum.astype(np.float32)
```
`.astype(np.float32)` converts from 64-bit to 32-bit floats.
Why? Storage efficiency: float32 uses half the memory of float64.
For our purposes (radio astronomy spectra), 32-bit precision is sufficient.
This halves our file sizes and network transfer times.

---

### collect_spectrum_batch() Method (Lines 408-448)

```python
def collect_spectrum_batch(self, num_averaged_spectra):
```
`num_averaged_spectra` will be `RHINOConfig.SPECTRA_PER_FILE = 60`.
This function collects 60 time-averaged spectra to make one file.

```python
spectra_batch = np.zeros((num_averaged_spectra, RHINOConfig.FFT_SIZE), 
                         dtype=np.float32)
```
Pre-allocates a 2D array to hold all spectra.
Shape: `(60, 16384)` — 60 rows (one per time-average), 16384 columns (one per frequency bin).

**Why pre-allocate?**
Alternative: start empty and append each spectrum.
Problem: Python must copy the entire array every time you append,
which is extremely slow for large arrays.
Pre-allocating once and filling in rows is much faster.

```python
batch_start = time.time()
```
Record start time — used to calculate progress and rate.

```python
for i in range(num_averaged_spectra):
    spectra_batch[i, :] = self.capture_time_averaged_spectrum()
```
`spectra_batch[i, :]` means: row `i`, all columns.
This assigns the returned array to the i-th row of the 2D array.

The loop runs 60 times. Each iteration:
1. Calls `capture_time_averaged_spectrum()` — runs for ~1 second
2. Stores the result in row `i` of the batch

Total time: ~60 seconds for a complete batch.

```python
if RHINOConfig.VERBOSE and (i+1) % 10 == 0:
```
`(i+1) % 10 == 0` uses the modulo operator `%`.
`%` gives the remainder after division.
`11 % 10 = 1`, `20 % 10 = 0`, `30 % 10 = 0` etc.
So this condition is True when `i+1` is divisible by 10:
at i=9 (10th), i=19 (20th), i=29 (30th), i=39 (40th), i=49 (50th), i=59 (60th).
In plain English: print progress every 10 spectra.

```python
rate = (i+1) / elapsed
eta = (num_averaged_spectra - i - 1) / rate if rate > 0 else 0
```
`rate` = spectra per second collected so far.
`eta` = estimated time remaining.
`(num_averaged_spectra - i - 1)` = how many spectra are left.
Dividing remaining spectra by the rate gives seconds remaining.
`if rate > 0 else 0` prevents dividing by zero at the very start.

---

### run_acquisition() Method (Lines 451-549)

This is the **main loop** — the engine of the whole script.

```python
if not self.pynq_available:
    print("[ERROR] Cannot run - PYNQ not available")
    return
```
`if not self.pynq_available:` — if PYNQ failed to import earlier, stop here.
`return` exits the function without running anything else.

```python
self.acquisition_start_time = time.time()
file_counter = 0
```
Record when acquisition started (used to calculate elapsed time).
Initialize file counter.

```python
try:
    while True:
```
`while True:` creates an **infinite loop** — it runs forever until
explicitly broken with `break` or until an exception is raised.
The `try:` means any exception will be caught below.

```python
elapsed = time.time() - self.acquisition_start_time
if elapsed >= RHINOConfig.ACQUISITION_DURATION:
    print("\n[ACQUISITION] Target duration reached")
    break
```
At the top of every loop iteration, check if we've exceeded the time limit.
`time.time()` − start time = seconds elapsed.
`break` exits the while loop cleanly.

```python
spectra_batch = self.collect_spectrum_batch(RHINOConfig.SPECTRA_PER_FILE)
```
Collect 60 time-averaged spectra (~60 seconds of data).

```python
timestamp = datetime.utcnow().isoformat()
filename = f"rhino_spectrum_{timestamp.replace(':', '-').replace('.', '_')}.npy"
```
`datetime.utcnow()` gets current UTC time (UTC = Coordinated Universal Time,
the same worldwide regardless of timezone — essential for astronomy).

`.isoformat()` formats it as: `"2026-02-11T14:30:00.123456"`

`.replace(':', '-').replace('.', '_')` makes it filename-safe
(filenames can't contain `:` or `.` reliably on all systems):
`"2026-02-11T14-30-00_123456"`

Final filename: `"rhino_spectrum_2026-02-11T14-30-00_123456.npy"`

```python
metadata = {
    'filename': filename,
    'timestamp': timestamp,
    ...
}
```
A Python **dictionary** — key-value pairs.
This records everything about how the data was collected so that
anyone loading the file later knows exactly how it was taken:
sample rate, FFT size, integration time, etc.
This is critically important for reproducible science!

```python
md5sum = hashlib.md5(spectra_batch).hexdigest()
metadata['md5sum'] = md5sum
```
`hashlib.md5(spectra_batch)` computes the MD5 hash of the raw bytes of the array.
`.hexdigest()` returns it as a readable hex string like `"a3f8d9e5c2b1f4e6..."`

This hash is stored in the metadata and sent with the data.
The server can recompute the hash and compare — if they match,
the data arrived intact. If not, there was a transmission error.

```python
if RHINOConfig.SAVE_LOCAL_COPY:
    import os
    os.makedirs(RHINOConfig.LOCAL_DATA_DIR, exist_ok=True)
```
`import os` here (not at the top) because it's only needed if local saving is enabled.
`os.makedirs(...)` creates the directory if it doesn't exist.
`exist_ok=True` means "don't raise an error if directory already exists".

```python
transmit_spectrum_data(spectra_batch, metadata)
```
Send the data to the logging server.
This calls the function defined earlier.

```python
file_counter += 1
self.total_files_transmitted += 1
```
Increment both counters (they track the same thing, used in different places).

```python
except KeyboardInterrupt:
    print("\n\n[ACQUISITION] Interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\n[ERROR] Acquisition failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    self._print_statistics()
```
`KeyboardInterrupt` is raised when you press Ctrl+C.
Catching it means we can print a clean exit message instead of an ugly crash.

`except Exception as e:` catches any unexpected error.
`traceback.print_exc()` prints the full error details including
which line caused the problem — essential for debugging.

`finally:` runs NO MATTER WHAT — whether there was an error, Ctrl+C,
or normal completion. Ensures statistics always get printed.

---

### _print_statistics() Method (Lines 552-570)

```python
data_mb = (self.total_files_transmitted * RHINOConfig.SPECTRA_PER_FILE * 
          RHINOConfig.FFT_SIZE * 4) / (1024*1024)
```
Calculates total data volume in MB:
- `total_files_transmitted`: number of files sent
- `SPECTRA_PER_FILE = 60`: spectra per file
- `FFT_SIZE = 16384`: values per spectrum
- `4`: bytes per float32 value
- `/ (1024*1024)`: converts bytes to megabytes

---


## SECTION 6: main() Function and Entry Point (Lines 605-649)

---

```python
def main():
```
This function is the entry point — the first function called when the script runs.

```python
if not (RHINOConfig.FFT_SIZE & (RHINOConfig.FFT_SIZE - 1) == 0):
```
This is a clever bit manipulation trick to check if a number is a power of 2.

How it works for 16384:
- 16384 in binary: `100000000000000`
- 16383 in binary: `011111111111111`
- 16384 & 16383 = `000000000000000` = 0

For any power of 2, `N & (N-1) == 0` is always True.
For any non-power-of-2, it's always False.
If it's not a power of 2, print error and exit.

```python
acquisition = RHINOAcquisition()
```
Creates an instance of the `RHINOAcquisition` class.
This triggers `__init__()` which:
1. Imports PYNQ
2. Loads the overlay
3. Gets hardware references
4. Calls `_configure_system()`

```python
acquisition.run_acquisition()
```
Starts the main acquisition loop.
This runs until:
- `ACQUISITION_DURATION` seconds have elapsed
- User presses Ctrl+C
- An error occurs

---

```python
if __name__ == '__main__':
    main()
```
This is one of the most important Python patterns.

When Python runs a file directly (`python3 script.py`),
it sets the special variable `__name__` to `'__main__'`.

When a file is imported by another file (`import script`),
`__name__` is set to `'script'` (the module name), NOT `'__main__'`.

This means:
- Running the script directly → `main()` is called → acquisition runs
- Importing the script as a library → `main()` is NOT called
  → can use the classes and functions without starting acquisition

This is considered best practice for all Python scripts.

---

## Summary: How All Pieces Connect

```
main()
  │
  ├─ Validate config
  │
  ├─ RHINOAcquisition()           ← Creates the acquisition object
  │     │
  │     ├─ Import PYNQ
  │     ├─ Load FPGA overlay
  │     ├─ Get hardware reference
  │     └─ _configure_system()    ← Set up hardware parameters
  │
  └─ acquisition.run_acquisition()
        │
        └─ while True:
              │
              ├─ collect_spectrum_batch(60)
              │     │
              │     └─ for 60 iterations:
              │           │
              │           └─ capture_time_averaged_spectrum()
              │                 │
              │                 └─ for 571 iterations:
              │                       │
              │                       └─ capture_single_spectrum()
              │                             │
              │                             └─ FPGA does FFT
              │                               (32 frames × 16384 pts)
              │
              ├─ Generate metadata dict
              ├─ Compute MD5 checksum
              │
              └─ transmit_spectrum_data()
                    │
                    ├─ Open WebSocket connection
                    ├─ Send metadata as JSON text
                    └─ send_array()
                          │
                          ├─ Pickle (serialize) array
                          └─ Send in 1 MB chunks
```

---

## Key Numbers to Remember

```
ADC sample rate:        300 MHz
Nyquist frequency:      150 MHz  (DC to 150 MHz coverage)
FFT size:               16384 bins
Frequency resolution:   18.3 kHz per bin

Hardware frames:        32 per spectrum
Time per spectrum:      1.75 ms

Software averaging:     571 spectra per integration
Integration time:       ~1 second

Spectra per file:       60 (1 per second = 1 minute of data per file)
File size:              60 × 16384 × 4 bytes = 3.9 MB per file
```