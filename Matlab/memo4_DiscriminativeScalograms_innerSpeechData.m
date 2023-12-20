load subject1   % Y= Y: class | session#
STs=permute(X,[2,3,1]);clear X; STs_baseline=permute(baseline,[2,3,1]); clear baseline
[Nsensors,Ntime,Ntrials]=size(STs); Fs=double(fs); time=[1:Ntime]*(1/Fs)
class_labels=Y(:,1)+1 % Class 0-->1 "shift one" upwards
session_labels=Y(:,2); clear Y
load sensor_xyz

%average re-ref
%re_STs=[];for i_trial=1:Ntrials, ST_DATA=STs(:,:,i_trial); re_STs(:,:,i_trial)=ST_DATA-mean(ST_DATA);end
%STs=re_STs;
%re_STs_baseline=[];for i_trial=1:3, ST_DATA=STs_baseline(:,:,i_trial); re_STs_baseline(:,:,i_trial)=ST_DATA-mean(ST_DATA);end
%STs_baseline=re_STs_baseline;


%%
% Task
tpre=knnsearch(time',0.5); tstart=knnsearch(time',1);tend=knnsearch(time',3.5); 

%% Step1 : constructing wavelet filterbanks
%[244 x 520 x 116]
fb = cwtfilterbank('SignalLength',Ntime,'SamplingFrequency',Fs,'FrequencyLimits',[4 40],'VoicesPerOctave',30);

%% Step2  : Estimating  single-trial (time-varying) CWT-transform & computing averaged Scalograms profiles
DiscrMAPS=[]; 
for i_sensor=1:Nsensors
    i_sensor/Nsensors
    WT=[];
    for i_trial=1:Ntrials
        signal = STs(i_sensor,:,i_trial);
        [wt,Faxis,coi]  = cwt(signal,'FilterBank',fb); % 100x1153 wavelet tranform of given trial
        WT(:,:,i_trial) = abs(wt); % Get absolute value since its imaginary number
    end

    sensorDiscrMaps=[];
    pair_no=0;
    for i1=1:3
        for i2=i1+1:4
            pair_no=pair_no+1;
            AAA1=WT(:,:,class_labels==i1); AA1=reshape(AAA1,[numel(Faxis)*Ntime,size(AAA1,3)])';         
            AAA2=WT(:,:,class_labels==i2); AA2=reshape(AAA2,[numel(Faxis)*Ntime,size(AAA2,3)])';
            paired_labels = [class_labels(class_labels==i1); class_labels(class_labels==i2)];
            [~, Z] = rankfeatures([AA1;AA2]',paired_labels,'criterion','ttest');
            sensorDiscrMaps(:,:,pair_no)=reshape(Z,numel(Faxis),Ntime);
        end
    end
    DiscrMAPS(:,:,:,i_sensor)=sensorDiscrMaps; 
end 




%% Step3 Presenting ta averaged Discriminant scalograms per sensor  (all pairs taken together) 

ALL_DiscrMAPS=squeeze(mean(DiscrMAPS,3));     % average across pairs; 
uplimit=max(ALL_DiscrMAPS(:));lolimit=min(ALL_DiscrMAPS(:));
figure(1),clf
for i_sensor=1:64, subplot(8,8,i_sensor)
surf(time,Faxis,(ALL_DiscrMAPS(:,:,i_sensor)),'edgecolor','none');title(sensor_names(i_sensor)) 
xline([1,3.5],'white','linewidth',1),xlabel('sec'),ylabel('Hz'),clim([lolimit,uplimit]),axis tight; view(0,90); end
colormap hot

figure(2),clf
for i_sensor=65:128, subplot(8,8,i_sensor-64)
surf(time,Faxis,(ALL_DiscrMAPS(:,:,i_sensor)),'edgecolor','none');title(sensor_names(i_sensor)) 
xline([1,3.5],'white','linewidth',1),xlabel('sec'),ylabel('Hz'),clim([lolimit,uplimit]),axis tight; view(0,90); end
colormap hot


%% Step4 presenting topographically the most informative sensors 
%a. averaged score within the action-interval 
figure(3),clf
Sensor_Score=[];for i_sensor=1:Nsensors,Sensor_Score(i_sensor)=mean(mean(ALL_DiscrMAPS(:,tstart:tend,i_sensor)));end
subplot(2,1,1),stem(Sensor_Score),xlabel('sensor #'),ylabel('Activation-score')
%threshold=quantile(Sensor_Score,.78);selected_sensor=find(Sensor_Score>threshold)
[~,list]=sort(Sensor_Score,'descend');slist=list(1:15) % 25 most discriminative sensors --> a design- parameter
subplot(2,1,2), plot(xyz(:,1),xyz(:,2),'ko'),hold on,plot(xyz(slist,1),xyz(slist,2),'r.','markersize',15)

%b. emphasizing the highest values within the action-interval 
figure(4),clf
Sensor_Score=[];for i_sensor=1:Nsensors,Q=ALL_DiscrMAPS(:,tstart:tend,i_sensor);
                    Sensor_Score(i_sensor)=quantile(Q(:),0.95);end
subplot(2,1,1),stem(Sensor_Score),xlabel('sensor #'),ylabel('Activation-score')
%threshold=quantile(Sensor_Score,.78);selected_sensor=find(Sensor_Score>threshold)
[~,list]=sort(Sensor_Score,'descend');slist2=list(1:25) % 25 most discriminative sensors  --> a design- parameter
subplot(2,1,2), plot(xyz(:,1),xyz(:,2),'ko'),hold on,plot(xyz(slist2,1),xyz(slist2,2),'r.','markersize',15)










