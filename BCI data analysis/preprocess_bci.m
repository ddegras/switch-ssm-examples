
function [y,S] = preprocess_bci(cnt,mrk)


% Number of time points in original data
T = size(cnt,1);

% Original sampling rate (Hz)
fs = 100;

% Stimulus sequence (-1:task 1, 0:rest, 1:task 2)
S = [];
if exist('mrk','var') 
    S = zeros(1,T);
    ntask = numel(mrk.pos); % number of tasks
    for i = 1:ntask
        idx = mrk.pos(i):min(mrk.pos(i)+4*fs,T); % task duration = 4s
        S(idx) = mrk.y(i);
    end 
    S = S + 2;
end

% Apply bandpass filtering
fpass = [8,25]; 
y = bandpass(double(cnt),fpass,fs);

% Downsample to 50Hz
if mod(T,2) == 1
    T = T - 1;
end 
odd = 1:2:T;
even = 2:2:T;
y = 0.5 * (y(odd,:) + y(even,:));
if ~isempty(S)
    S = S(odd);
end

% Standardize, remove extreme observations, and re-standardize
y = (y - mean(y)) ./ std(y);
th = 10; % threshold
y(y > th) = th; 
y(y < -th) = -th; 
y = (y - mean(y)) ./ std(y);
y = y';

