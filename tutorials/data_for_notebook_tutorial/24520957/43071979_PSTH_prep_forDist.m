% %Script to pre-process data for LFADS 
% These are data files of a precision center out task from the Rouse Precision Neural Dynamics Lab/Schieber Finger Movement Lab.
% This is data that has been pre-processed to be submitted to LFADS. Data for animal P accompany the Pytorch LFADS tutorial available that is available at https://github.com/arsedler9/lfads-torch
% The original data is available here without the pre-processing is available: https://doi.org/10.6084/m9.figshare.23631951
% The processing script PSTH_prep_forDist.m shared here converts the original data into initial and corrective submovements.
% 
% The data was aligned to the peak velocity of movement for this analysis.  All variables with *_peakVel are time aligned from 400ms before peak velocity until 200ms after in 20ms time steps. Sample 21 is the sample when peak velocity occured.
% 
% The data variables are as follows:
% spikes_peakVel: Neural data for each submovement aligned to peak velocity binned in 20 ms bins, spiking unit x time x submovement
% JoystickPos_peakVel: Position of the joystick and resulting cursor on the display,  x&y x time x submovement
% 
% Information about the submovements aligned to peak velocity are available in the following variables:
% timeMs_peakVel  : array with time in ms relative to peak velocity = 0
% 
% %The submovments are defined by the cursor movement from 100ms before until 100ms after peak velocity and represented with a straight-line vector
% %The start and end of the vector are defined by JoystickPos_peakVel(:, [start_behav_samples_peakVel=16,end_behav_samples_peakVel=26],:)
% direction_deg_peakVel:  Direction in degrees of movement, straight right (+x) is 0 degrees, straight up (+y) is 90 degrees
% magnitude_peakVel:   Magnitude of the movement from 100ms before until 100ms after peak velocity
% 
% peakVel_tr_label: trial labels that allow identifying which original trial each submovement came from
% conditionID_peakVel: Trial condition for given submovement. IDs 1-24 are for intiial amd 25-32 for corrective submovements
% %conditionID_peakVel 1-8 the initial movements to 8 regular sized targets counterclockwise (1 to right, 8 down-right)
% %conditionID_peakVel 9-16 the initial movements to 8 narrow sized targets counterclockwise (9 to right, 16 down-right)
% %conditionID_peakVel 17-24 the initial movements to 8 shallow sized targets counterclockwise (17 to right, 24 down-right)
% %conditionID_peakVel 25-32 the corrective submovements binned by direction counterclockwise (25 to right, 32 down-right)
% targetID_peakVel: Target for current submovement
% %conditionID_peakVel 1-8 for the 8 regular sized targets counterclockwise (1 to right, 8 down-right)
% %conditionID_peakVel 9-16 for the 8 narrow sized targets counterclockwise (9 to right, 16 down-right)
% %conditionID_peakVel 17-24 for the 8 shallow sized targets counterclockwise (17 to right, 24 down-right)
% 
% Other variables not directly corresponding to submovements
% condition_psth: mean of spikes_peakVel for each of the 32 conditions in conditionID_peakVel, condition x time x spiking unit 
% dataMask_peakVel: data mask used by getMaskedData to get data time-aligned to peak velocity from original trial data
% 
% JoystickPos_disp: Original joystick for each trial x&y x time x trial
% targetID: target for each original trial, 1-8 for regular, 9-16 for narrow, and 17-24 for shallow targets  
% timeMs: time in ms for the original trial aligned data samples

%WeiHsien Lee and Adam Rouse, Precision Neural Dynamics Lab
% University of Kansas Medical Center
% 2023/11/7

%% Data import
clc;clear;close all;
Data = 'P_Spikes_20170630-data';
remove_highly_correlated_channels = 1;  %This step can be memory intensive 
animal = Data(1); %Valid for animals P and Q

data_path = 'C:\dataLocation\';  %Change to data path of original data
%Data is available here: https://doi.org/10.6084/m9.figshare.23631951

savedir = ['Animal' animal];
if ~exist(savedir, 'dir')
    mkdir(savedir)
end

% addpath('D:\documents\2020 summer-LFAD\pndl-matlab-tools-main');%add lab function [file locations path], could hard code that
load([data_path Data '.mat'],'SpikeTimes','SpikeSettings','SpikeQuality', 'TrialInfo','PeakInfo', 'TrialSettings', 'JoystickPos_disp', 'CursorSpeedRaw')
%file_location change to specified data location
%% resort SpikeTimes relavant to trial start time
% Only include M1 arrays
if animal == 'P'
    arraysToInclude = {'G', 'H', 'I', 'J', 'K', 'L'};
elseif animal == 'Q'
    arraysToInclude = {'H', 'I', 'J', 'K', 'alpha'};    
%     arraysToInclude = {'H'};
elseif animal == 'H'
    arraysToInclude = {'H'};
end
%% Keep only channels from arraysToInclude
channelsIncluded = ismember(SpikeSettings.array_by_channel, arraysToInclude);
SpikeTimes = SpikeTimes(channelsIncluded,:);
%% extract max/min spiketime
for jj = 1:length(SpikeTimes)
    for ii = 1:length(TrialInfo.trial_start_time)
        if isempty(SpikeTimes{jj,1}{1,ii}) == 0
            max_spiketime_mat{jj,1}(1,ii) = max(SpikeTimes{jj,1}{1,ii});
            min_spiketime_mat{jj,1}(1,ii) = min(SpikeTimes{jj,1}{1,ii});
        else
            max_spiketime_mat{jj,1}(1,ii) = 0;
            min_spiketime_mat{jj,1}(1,ii) = 0; 
        end
    end
    max_spiketime(jj,1) = max(max_spiketime_mat{jj,1});
    min_spiketime(jj,1) = min(min_spiketime_mat{jj,1});
 end
max_SpikeTime = max(max_spiketime);
min_SpikeTime = min(min_spiketime);
%% Config
bin=1;  %10 ms, 100 Hz
timeMs = min_SpikeTime*1000:bin:max_SpikeTime*1000; %in Ms
timeWindowPre = timeMs(1);
timeWindowPost = timeMs(end);
timeMs = timeMs.';
% sizes in different dimensions
num_neurons = size(SpikeTimes,1);
num_trials = size(SpikeTimes{1},2);

trial_histcounts = zeros(length(timeMs)-1, num_trials, num_neurons);
for n = 1:num_neurons
    for trial = 1:num_trials
        trial_histcounts(:,trial,n) = histcounts(1000*SpikeTimes{n}{trial}, timeMs) / (bin/1000);
    end
end
trial_histcounts = permute(trial_histcounts,[2 1 3]);%sort as trial x time x neuron for peak alignment
spikes=permute(trial_histcounts,[3 2 1]);%sort as neuron x time x trial for LFADS

cat_spikes = reshape(spikes, size(spikes,1),[]);
c = corr(cat_spikes'); % check for memory issues
c(c==1) = 0;
figure
edges=(0.001:0.005:0.95);
histogram(c(:),edges);
ylim([0 5])
cutoff=0.02;
title('total coeff in all pairs')
xlabel('coef')
ylabel('counts')
%% excluding channels with higher correlation by histogram/spike_ranks
%find max coorrlelated channels
for ch = 1:size(c,1)
    max_c(ch) = max([c(ch,:),c(:,ch)']);
end
figure
plot(max_c);
cutoff=0.02; %set manually base on plot
title('max coeff over channels')
xlabel('channel')
ylabel('coef')
channels_corr = find(c>cutoff);
%channels_corr=[6 14 16 34 61 62 68 85 86 89 84];
%channelsIncluded(channels_corr)=0;
%% bin back to 10/20 then calculate psth // could put this part below as another script to reduce redundency
%new spikes
SpikeQuality.corr_matrix = c;%for M1 neurons
SpikeQuality.high_corr2remove=[];
% SpikeQuality.high_corr_over_threshold=;
% SpikeQuality.highest_corr_ch=;
spike_count = sum(cat_spikes,2);
threshold = 0.1;
all_corr_ch=[];
while max(c(:))>threshold
    [x, y]=find(c==max(c(:)));
    if spike_count(x(1))>spike_count(y(1))
        tmp =y(1);
        SpikeQuality.high_corr2remove = [SpikeQuality.high_corr2remove tmp];
    else
        tmp =x(1);
        SpikeQuality.high_corr2remove = [SpikeQuality.high_corr2remove tmp];
    end
    c(tmp,:)=0;
    c(:,tmp)=0;
    all_corr_ch=[all_corr_ch x(1) y(1)];
end
SpikeQuality.all_corr_ch = unique(all_corr_ch);

%% bin back to 10/20 then calculate psth // could put this part below as another script to reduce redundency
%new spikes
% Config
bin=20;
timeMs = min_SpikeTime*1000:bin:max_SpikeTime*1000; %in Ms
timeWindowPre = timeMs(1);
timeWindowPost = timeMs(end);
timeMs = timeMs.';
% sizes in different dimensions
num_neurons = size(SpikeTimes,1);
num_times = length(timeMs)-1; 
num_trials = size(SpikeTimes{1},2);
targetID = TrialInfo.trial_target;
%% calculate trial PSTH
trial_histcounts = nan(length(timeMs)-1, num_trials, num_neurons);
for n = 1:num_neurons
    for trial = 1:num_trials
        trial_histcounts(:,trial,n) = histcounts(1000*SpikeTimes{n}{trial}, timeMs) / (bin/1000);
        %
        trial_histcounts( timeMs(2:end)>(1000*TrialInfo.align_times_plx(trial,6)) ,trial,n) = NaN;
%         if ~isempty(SpikeTimes{n}{trial}) %check for max spike time
%         max_spike_time(trial,n) = max(SpikeTimes{n}{trial});
%         else
%             max_spike_time(trial,n) = 0;
%         end
    end
    
end
trial_histcounts = permute(trial_histcounts,[2 1 3]);%sort as trial x time x neuron for peak alignment
%generate datsmask for peak speed alignment 
dataMask_peakVel = zeros(size(PeakInfo.speedPeaksTroughs_i,1),size(trial_histcounts,3)); %tr*t
speedPeaksInd = round(PeakInfo.speedPeaksTroughs_i(:,2)*(10/bin));  %Speed peak indexes 
peak_before_samp = -round(400/bin);
peak_after_samp = round(200/bin)-1;
for tr = 1:size(PeakInfo.speedPeaksTroughs_i,1)
     curr_indexes = speedPeaksInd(tr) + (peak_before_samp:peak_after_samp);
     curr_indexes = curr_indexes(curr_indexes>0);
     dataMask_peakVel(tr,curr_indexes) = 1;
end
timeMs_peakVel = (peak_before_samp*bin):bin:(peak_after_samp*bin);
%create a data mask for chop data  %TODO - eventually will go back to trying to use the random chops
%assign random start point between 1:200ms(1-21 in timeMS)
spikes = permute(trial_histcounts,[3 2 1]);%sort as neuron x time x trial for LFADS
rng('default');
t0_mat = randi([1 200/bin+1],1,size(spikes,3));
num_chops_per_trial = floor((size(spikes,2)-(200/bin+1))/(400/bin));
dataMask_chop_cat = zeros(num_chops_per_trial*num_trials, num_times);
chop_tr_label = zeros(num_chops_per_trial*num_trials,1);
chop_start_index = zeros(num_chops_per_trial*num_trials,1);
for tr=1:num_trials
    for cp=1:num_chops_per_trial
        curr_indexes = t0_mat(tr)+(0:(600/bin-1))+(600*2/(bin*3))*(cp-1);
        if curr_indexes(end)<=num_times
        chop_start_index((tr-1)*num_chops_per_trial + cp) = curr_indexes(1);
        dataMask_chop_cat((tr-1)*num_chops_per_trial + cp, curr_indexes) = 1;
        chop_tr_label((tr-1)*num_chops_per_trial + cp, 1) = tr;
        else
        chop_tr_label((tr-1)*num_chops_per_trial + cp, 1) = tr;
        end
    end  
end
%analyze behavior
%use time window from 100 ms before to 100ms after speed peaks to characterize direction and magnitude of reach
start_behav_samples_peakVel = -peak_before_samp - 100/bin +1; 
end_behav_samples_peakVel  = -peak_before_samp + 100/bin +1;

if bin > 10 && mod(bin,10)==0
    tmp_pos = nan(size(JoystickPos_disp,1), num_times, 2);
    tmp_speed = nan(size(JoystickPos_disp,1), num_times);
    for tr = 1:size(JoystickPos_disp,1)
        for d = 1:2
            tmp_input = JoystickPos_disp(tr,:,d);
            tmp_input = tmp_input(~isnan(tmp_input));
            tmp_pos(tr,1:ceil(length(tmp_input)/(bin/10)),d) = decimate(tmp_input, bin/10);
        end
        tmp_input = CursorSpeedRaw(tr,:);
        tmp_input = tmp_input(~isnan(tmp_input));
        tmp_speed(tr,1:ceil(length(tmp_input)/(bin/10))) = decimate(tmp_input, bin/10);
    end
    JoystickPos_disp = tmp_pos;
    CursorSpeedRaw = tmp_speed;
end

JoystickPos_peakVel = getMaskedData(JoystickPos_disp, dataMask_peakVel, PeakInfo.trial_ids_peakVel);
CursorSpeedRaw_peakVel = getMaskedData(CursorSpeedRaw, dataMask_peakVel, PeakInfo.trial_ids_peakVel);
Dir=JoystickPos_peakVel(:, [start_behav_samples_peakVel,end_behav_samples_peakVel],:);
degreecutoff=(-22.5:45:(360-22.5)); %reach direction cutoffs 
for i=1:length(PeakInfo.trial_ids_peakVel)
    avgslope=atan2((Dir(i,end,2)-Dir(i,1,2)),(Dir(i,end,1)-Dir(i,1,1)));
    direction_deg_peakVel(i)=rad2deg(avgslope);
    mag=sqrt((Dir(i,end,2)-Dir(i,1,2))^2+(Dir(i,end,1)-Dir(i,1,1))^2);
    magnitude_peakVel(i)=mag;
    if direction_deg_peakVel(i)<-22.5
        direction_deg_peakVel(i)=direction_deg_peakVel(i)+360;
    end
end
for i=1:length(PeakInfo.trial_ids_peakVel)
    for j=1:length(degreecutoff)-1
        if direction_deg_peakVel(i)>=degreecutoff(j) && direction_deg_peakVel(i)<degreecutoff(j+1) 
                directionID_peakVel(i)=j;  %reach direction ID (1 to right, 8 down-right) 
        end
    end   
end
spikes_peakVel = getMaskedData(trial_histcounts, dataMask_peakVel, PeakInfo.trial_ids_peakVel);
spikes_peakVel = permute(spikes_peakVel,[3 2 1]);
targetID_peakVel = targetID(PeakInfo.trial_ids_peakVel);

%Create conditionID_peakVel - a condition ID for the initial and corrective submomements (the
%peakVel variables)
%conditionID_peakVel 1-8 the initial movements to 8 regular sized targets counterclockwise (1 to right, 8 down-right)
%conditionID_peakVel 9-16 the initial movements to 8 narrow sized targets counterclockwise (9 to right, 16 down-right)
%conditionID_peakVel 17-24 the initial movements to 8 shallow sized targets counterclockwise (17 to right, 24 down-right)
%conditionID_peakVel 25-32 the corrective submovements binned by direction counterclockwise (25 to right, 32 down-right)
conditionID_peakVel = zeros(size(targetID_peakVel));
for c = 1:24
    conditionID_peakVel( targetID_peakVel == c & PeakInfo.initPeak_flag) = c;
end
for c = 1:8
    conditionID_peakVel( directionID_peakVel' == c & ~PeakInfo.initPeak_flag ) = 24+c;
end

%assign psth to each conditionID
condition_psth = zeros(num_neurons, size(spikes_peakVel,2), 32);
for n = 1:num_neurons
    for c = 1:max(conditionID_peakVel)
    condition_psth(n,:,c) = mean(spikes_peakVel(n,:,conditionID_peakVel==c),3);
    end
 end
%%
spikes_chopped = getMaskedData(permute(spikes, [3,2,1]), dataMask_chop_cat, chop_tr_label );
nan_chops = any(any(isnan(spikes_chopped),2),3);%
spikes_chopped = spikes_chopped(~nan_chops,:,:);
spikes_chopped = permute(spikes_chopped,[3,2,1]);%n x t x tr
dataMask_chop_cat = dataMask_chop_cat(~nan_chops,:);
chop_tr_label = chop_tr_label(~nan_chops);
chop_start_index = chop_start_index(~nan_chops);
targetID_peakVel = targetID(PeakInfo.trial_ids_peakVel);
condition_psth = permute(condition_psth,[3 2 1]);%sorted for LFADS
JoystickPos_disp = permute(JoystickPos_disp,[3,2,1]); %x/y x t x tr
JoystickPos_peakVel = permute(JoystickPos_peakVel, [3,2,1]); %x/y x t x tr
peakVel_tr_label = PeakInfo.trial_ids_peakVel;



if remove_highly_correlated_channels
    inds_to_keep = setdiff(1:1:size(spikes_peakVel, 1), SpikeQuality.high_corr2remove);
    spikes_peakVel = spikes_peakVel(inds_to_keep, :, :);
end 


save([savedir Data '_PSTH_prep_bin_' num2str(bin) '.mat'],'JoystickPos_disp','targetID','timeMs', 'timeMs_peakVel', ...
    'spikes_peakVel','condition_psth','conditionID_peakVel', 'targetID_peakVel','dataMask_peakVel','peakVel_tr_label','JoystickPos_peakVel','direction_deg_peakVel','magnitude_peakVel')

