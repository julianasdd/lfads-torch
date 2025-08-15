These are data files of a precision center out task from the Rouse Precision Neural Dynamics Lab/Schieber Finger Movement Lab.
This is data that has been pre-processed to be submitted to LFADS. Data for animal P accompany the Pytorch implementation of LFADS tutorial  available at https://github.com/arsedler9/lfads-torch
The original data is available here without the pre-processing is available: https://doi.org/10.6084/m9.figshare.23631951.v1 
The processing script PSTH_prep_forDist.m shared here converts the original data into initial and corrective submovements.

The data was aligned to the peak velocity of movement for this analysis.  All variables with *_peakVel are time aligned from 400ms before peak velocity until 200ms after in 20ms time steps. Sample 21 is the sample when peak velocity occured.

The data variables are as follows:
spikes_peakVel: Neural data for each submovement aligned to peak velocity binned in 20 ms bins, spiking unit x time x submovement
JoystickPos_peakVel: Position of the joystick and resulting cursor on the display,  x&y x time x submovement

Information about the submovements aligned to peak velocity are available in the following variables:
timeMs_peakVel  : array with time in ms relative to peak velocity = 0

%The submovments are defined by the cursor movement from 100ms before until 100ms after peak velocity and represented with a straight-line vector
%The start and end of the vector are defined by JoystickPos_peakVel(:, [start_behav_samples_peakVel=16,end_behav_samples_peakVel=26],:)
direction_deg_peakVel:  Direction in degrees of movement, straight right (+x) is 0 degrees, straight up (+y) is 90 degrees
magnitude_peakVel:   Magnitude of the movement from 100ms before until 100ms after peak velocity

peakVel_tr_label: trial labels that allow identifying which original trial each submovement came from
conditionID_peakVel: Trial condition for given submovement. IDs 1-24 are for intiial amd 25-32 for corrective submovements
%conditionID_peakVel 1-8 the initial movements to 8 regular sized targets counterclockwise (1 to right, 8 down-right)
%conditionID_peakVel 9-16 the initial movements to 8 narrow sized targets counterclockwise (9 to right, 16 down-right)
%conditionID_peakVel 17-24 the initial movements to 8 shallow sized targets counterclockwise (17 to right, 24 down-right)
%conditionID_peakVel 25-32 the corrective submovements binned by direction counterclockwise (25 to right, 32 down-right)
targetID_peakVel: Target for current submovement
%conditionID_peakVel 1-8 for the 8 regular sized targets counterclockwise (1 to right, 8 down-right)
%conditionID_peakVel 9-16 for the 8 narrow sized targets counterclockwise (9 to right, 16 down-right)
%conditionID_peakVel 17-24 for the 8 shallow sized targets counterclockwise (17 to right, 24 down-right)

Other variables not directly corresponding to submovements
condition_psth: mean of spikes_peakVel for each of the 32 conditions in conditionID_peakVel, condition x time x spiking unit 
dataMask_peakVel: data mask used by getMaskedData to get data time-aligned to peak velocity from original trial data

JoystickPos_disp: Original joystick for each trial x&y x time x trial
targetID: target for each original trial, 1-8 for regular, 9-16 for narrow, and 17-24 for shallow targets  
timeMs: time in ms for the original trial aligned data samples

