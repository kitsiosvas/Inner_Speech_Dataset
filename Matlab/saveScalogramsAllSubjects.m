% Saves all 10 subjects scalograms in a .mat file. The outpout file has size [100x1153x128x10] and it's
% size is ~1Gb

scalograms = [];

for iSubject=1:10

    filename = sprintf("subject%d", iSubject);
    load(filename)

    STs=permute(X,[2,3,1]);clear X; STs_baseline=permute(baseline,[2,3,1]); clear baseline
    [Nsensors,Ntime,Ntrials]=size(STs); Fs=double(fs); time=[1:Ntime]*(1/Fs);
    class_labels=Y(:,1)+1; % Class 0-->1 "shift one" upwards
    session_labels=Y(:,2); clear Y
    load sensor_xyz
    
    
    %%
    % Task
    tpre=knnsearch(time',0.5); tstart=knnsearch(time',1);tend=knnsearch(time',3.5); 
    
    %% Step1 : constructing wavelet filterbanks
    %[244 x 520 x 116]
    fb = cwtfilterbank('SignalLength',Ntime,'SamplingFrequency',Fs,'FrequencyLimits',[4 40],'VoicesPerOctave',30);
    
    %% Step2  : Estimating  single-trial (time-varying) CWT-transform & computing averaged Scalograms profiles
    sensorSCAL=[]; 
    for i_sensor=1:Nsensors
        i_sensor/Nsensors
    
        % Calculate the wavelet transform (100x1153) of the i-th sensor across ALL trials
        % single trial: 1x1153 --> wavelet transform: 100x1153 --> for all trials: 100x1153x200
        WT=[];
        for i_trial=1:Ntrials
            signal = STs(i_sensor,:,i_trial); 
            [wt, Faxis, coi] = cwt(signal,'FilterBank',fb);
            WT(:,:,i_trial) = wt;
        end
        % Get the average for each class: 100x1153x200 --> 100x1153 4 times (one per class)
        SCAL1 = mean(abs(WT), 3);

        % Save the 4 scalograms to sensorSCAL variable for each sensor: 100x1153x128x4
        sensorSCAL(:,:,i_sensor) = SCAL1;  
    end

    scalograms(:, :, :, iSubject) = sensorSCAL;
end


save('scalogramsAllSubjects.mat', 'scalograms')

