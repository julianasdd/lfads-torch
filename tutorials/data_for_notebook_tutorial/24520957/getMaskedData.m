
function [alignedData, median_t] = getMaskedData(data, dataMask, trialNum, nanReversePadding)
n_signals = size(data,3);
%nanReversePadding- if true the first columns are filled with nan's rather
%than the typical nan padding at the end
if nargin<4
    nanReversePadding = false;
end
if nargin<3 || isempty(trialNum)
    if max(dataMask(:)) == 1  %If logicals or 0/1 numbers
        n_tr = size(dataMask,1);
        n_samps = sum(logical(dataMask),2);
        max_t = max(n_samps);
        alignedData = nan(n_tr, max_t, n_signals);
        for tr = 1:n_tr
            if ~nanReversePadding
                alignedData(tr,1:n_samps(tr),:) = data(tr, logical(dataMask(tr,:)), :);
            else
                alignedData(tr,(end-n_samps(tr)+1):end,:) = data(tr, logical(dataMask(tr,:)), :);
            end
        end
    else
        trialNum = unique(dataMask(:));
        trialNum = trialNum(trialNum~=0);
        n_tr = length(trialNum);
        max_t = sum(dataMask(:)==mode(dataMask(dataMask~=0)));
        alignedData = nan(n_tr, max_t, n_signals);
        n_samps = zeros(n_tr,1);
        for tr = 1:n_tr
            [n_samps(tr), curr_tr] = max(sum(dataMask == trialNum(tr),2)); %Warning: only one trial can have indexes
if ~nanReversePadding
            alignedData(tr,1:n_samps(tr),:) = data(curr_tr, dataMask(curr_tr,:) == trialNum(tr), :);
else
alignedData(tr,(end-n_samps(tr)+1):end,:) = data(curr_tr, dataMask(curr_tr,:) == trialNum(tr), :);
end
        end
    end
else
    n_tr = length(trialNum);
    n_samps = sum(logical(dataMask),2);
    max_t = max(n_samps);
    alignedData = nan(n_tr, max_t, n_signals);
    for tr = 1:n_tr
        if ~nanReversePadding
            alignedData(tr,1:n_samps(tr),:) = data(trialNum(tr), logical(dataMask(tr,:)), :);
        else
            alignedData(tr,(end-n_samps(tr)+1):end,:) = data(trialNum(tr), logical(dataMask(tr,:)), :);
        end
    end
end
median_t = median(n_samps);
end