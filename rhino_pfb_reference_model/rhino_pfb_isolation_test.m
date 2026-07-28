%% RHINO custom overlay - adjacent-channel isolation test
%
% Tests something more realistic than a single on-bin tone: a strong
% signal alongside a much weaker one, to see how close the weak signal
% can sit before the strong one's own leakage swamps it. This is the
% same figure of merit behind the project's established PFB-vs-FFT
% adjacent-channel isolation comparisons.
%
% Window: Blackman applied to a sinc kernel, matching the actual RHINO
% PFB configuration (rhino-daq/src/pfb_funcs.py create_window(), and
% rhino-daq/obs_config.yaml: pfbParams.appliedWindow = blackman).
% blackman(L,'periodic') matches scipy's get_window() default convention.
%
% Verified in Python first: weak tone (-60 dB) one bin from a strong
% tone measured at -38.36 dB (masked by the strong tone's own leakage);
% the same weak tone five bins away measured -60.01 dB (cleanly
% resolved). Running this script should reproduce those same values.

clear; clc;

N = 16384; P = 4; L = N*P;
n = (0:L-1)' - (L-1)/2;
h_sinc = sinc(n/N);
h_window = blackman(L, 'periodic');
h_proto = h_sinc .* h_window;
h_proto = h_proto / sum(h_proto);
h_poly = reshape(h_proto, N, P);

num_frames_needed = 3;
total_len = N * (P - 1 + num_frames_needed);
t = (0:total_len-1)';

strong_bin = 100;
weak_true_db = -60;
weak_amp = 10^(weak_true_db/20);

%% Case 1: weak tone immediately adjacent (1 bin away)
weak_bin_near = strong_bin + 1;
x_near = cos(2*pi*strong_bin*t/N) + weak_amp*cos(2*pi*weak_bin_near*t/N);
spec_near = pfb_channelize(x_near, h_poly, N, P);
mag_near_db = 20*log10(abs(spec_near(end,:))/max(abs(spec_near(end,:))));

fprintf('--- Weak tone 1 bin away ---\n');
fprintf('True level: %.2f dB, measured: %.2f dB (error: %.2f dB)\n', ...
    weak_true_db, mag_near_db(weak_bin_near+1), ...
    mag_near_db(weak_bin_near+1) - weak_true_db);

%% Case 2: weak tone 5 bins away
weak_bin_far = strong_bin + 5;
x_far = cos(2*pi*strong_bin*t/N) + weak_amp*cos(2*pi*weak_bin_far*t/N);
spec_far = pfb_channelize(x_far, h_poly, N, P);
mag_far_db = 20*log10(abs(spec_far(end,:))/max(abs(spec_far(end,:))));

fprintf('\n--- Weak tone 5 bins away ---\n');
fprintf('True level: %.2f dB, measured: %.2f dB (error: %.2f dB)\n', ...
    weak_true_db, mag_far_db(weak_bin_far+1), ...
    mag_far_db(weak_bin_far+1) - weak_true_db);

%% Plot both cases side by side
figure;
subplot(1,2,1);
plot(strong_bin-5:strong_bin+15, mag_near_db(strong_bin-4:strong_bin+16));
title('Weak tone 1 bin away (Blackman)'); xlabel('Channel (bin)'); ylabel('dB');
grid on;
subplot(1,2,2);
plot(strong_bin-5:strong_bin+15, mag_far_db(strong_bin-4:strong_bin+16));
title('Weak tone 5 bins away (Blackman)'); xlabel('Channel (bin)'); ylabel('dB');
grid on;

function spec = pfb_channelize(x, h_poly, N, P)
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
