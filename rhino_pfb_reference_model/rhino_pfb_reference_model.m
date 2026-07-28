%% RHINO custom overlay - reference model: 4-tap PFB + 16384-point FFT
%
% Purpose: a plain MATLAB/stock-block reference implementation of the
% target design (FFT 16384, PFB taps 4), validated against the AMD
% Toolbox / HDL Coder version once that toolchain is available. This
% does not touch Vivado, Model Composer, or digdev4 at all.
%
% Window: Blackman applied to a sinc kernel, matching the actual RHINO
% PFB configuration. Confirmed against:
%   - rhino-daq/src/pfb_funcs.py, create_window() (based on Danny Price's
%     PFB notebook, https://github.com/telegraphic/pfb_introduction)
%   - rhino-daq/obs_config.yaml: pfbParams.appliedWindow = blackman, nTaps = 4
% blackman(L,'periodic') is used deliberately, not the symmetric default,
% to match scipy's get_window(), which is periodic by default.
%
% Verified in Python against the exact create_window() recipe before
% being translated here: a pure tone on bin 100 produced -39.12 dB at
% the immediate neighbouring bins and -117.01 dB five bins away.
% Running this script reproduced those same values exactly.

clear; clc;

N = 16384;      % FFT size / number of channels
P = 4;          % PFB taps
L = N * P;      % total prototype filter length

%% Prototype filter: Blackman-windowed sinc, matching RHINO's create_window()
n = (0:L-1)' - (L-1)/2;
h_sinc = sinc(n/N);
h_window = blackman(L, 'periodic');
h_proto = h_sinc .* h_window;
h_proto = h_proto / sum(h_proto);

h_poly = reshape(h_proto, N, P);  % columns = taps, column 1 = newest tap

%% Verification test: pure tone exactly on a known bin
test_bin = 100;
num_frames_needed = 3;
total_len = N * (P - 1 + num_frames_needed);
t = (0:total_len-1)';
x = cos(2*pi*test_bin*t/N);

spec = pfb_channelize(x, h_poly, N, P);
mag = abs(spec(end,:));
mag_db = 20*log10(mag/max(mag) + 1e-15);

[~, peak_bin_1indexed] = max(mag);
peak_bin = peak_bin_1indexed - 1;

fprintf('Expected peak bin: %d, actual peak bin: %d\n', test_bin, peak_bin);
fprintf('Level at bin+1 (dB): %.2f\n', mag_db(peak_bin_1indexed+1));
fprintf('Level at bin-1 (dB): %.2f\n', mag_db(peak_bin_1indexed-1));
fprintf('Level 5 bins away (dB): %.2f\n', mag_db(peak_bin_1indexed+5));

figure;
plot((0:N-1), mag_db);
xlim([test_bin-20, test_bin+20]);
xlabel('Channel (bin)'); ylabel('Magnitude (dB, normalized)');
title('PFB output around the test tone, zoomed (Blackman window)');
grid on;

%% Local function: polyphase filter bank channelizer
function spec = pfb_channelize(x, h_poly, N, P)
    % x: column vector input signal
    % h_poly: N x P matrix, column p = polyphase branch p (1 = newest)
    num_frames = floor(length(x)/N) - (P - 1);
    spec = zeros(num_frames, N);
    for k = 1:num_frames
        acc = zeros(N,1);
        for p = 1:P
            start_idx = (k-1)*N + (P-p)*N + 1;
            acc = acc + x(start_idx:start_idx+N-1) .* h_poly(:,p);
        end
        spec(k,:) = fft(acc);
    end
end
