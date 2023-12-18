load subject1   % Y= Y: class | session#
STs=permute(X,[2,3,1]);clear X; STs_baseline=permute(baseline,[2,3,1]); clear baseline
[Nsensors,Ntime,Ntrials]=size(STs); Fs=double(fs); time=[1:Ntime]*(1/Fs);
class_labels=Y(:,1)+1; % Class 0-->1 "shift one" upwards
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
sensorSCAL=[]; 
for i_sensor=1:Nsensors
    i_sensor/Nsensors

    WT=[];
    for i_trial=1:Ntrials
        signal = STs(i_sensor,:,i_trial); 
        [wt, Faxis, coi] = cwt(signal,'FilterBank',fb);
        WT(:,:,i_trial) = wt;
    end
     SCAL1 = mean(abs(WT(:,:,class_labels==1)), 3);
     SCAL2 = mean(abs(WT(:,:,class_labels==2)), 3);
     SCAL3 = mean(abs(WT(:,:,class_labels==3)), 3);
     SCAL4 = mean(abs(WT(:,:,class_labels==4)), 3);
     sensorSCAL(:,:,i_sensor, 1) = SCAL1;
     sensorSCAL(:,:,i_sensor, 2) = SCAL2;
     sensorSCAL(:,:,i_sensor, 3) = SCAL3;
     sensorSCAL(:,:,i_sensor, 4) = SCAL4;
end   

%% Step3 Presenting ta averaged scalograms per sensor  (all 4 classes taken together) 

ALL_sensorSCAL=mean(sensorSCAL,4);
uplimit = max(ALL_sensorSCAL(:));
lolimit = min(ALL_sensorSCAL(:));
figure(1),clf
for i_sensor=1:64, subplot(8,8,i_sensor)
surf(time,Faxis,squeeze(ALL_sensorSCAL(:,:,i_sensor)),'edgecolor','none');title(sensor_names2(i_sensor)) 
xline([1,3.5],'white','linewidth',1),xlabel('sec'),ylabel('Hz'),clim([lolimit,uplimit]),axis tight; view(0,90); end

figure(2),clf
for i_sensor=65:128, subplot(8,8,i_sensor-64)
surf(time,Faxis,squeeze(ALL_sensorSCAL(:,:,i_sensor)),'edgecolor','none');title(sensor_names2(i_sensor)) 
xline([1,3.5],'white','linewidth',1),xlabel('sec'),ylabel('Hz'),clim([lolimit,uplimit]),axis tight; view(0,90); end


%% Step4 presenting the Relative Change with respect to [0 0.5]sec period 
   % based on the previous averaged scalograms per sensor  (all 4 classes taken together) 

% difference from baseline [0 0.5]sec
%DAA=ALL_sensorSCAL-(mean(ALL_sensorSCAL(:,1:tpre,:),2)); limit=max(abs(DAA(:)));
%DAA=(ALL_sensorSCAL-(mean(ALL_sensorSCAL(:,1:tpre,:),2)))./(std(ALL_sensorSCAL(:,1:tpre,:),[],2)); limit=5 %limit=max(abs(DAA(:)));
DAA=(ALL_sensorSCAL-(mean(ALL_sensorSCAL(:,1:tpre,:),2)))./(mean(ALL_sensorSCAL(:,1:tpre,:),2));  limit=max(abs(DAA(:)));

figure(3),clf
for i_sensor=1:64, subplot(8,8,i_sensor)
surf(time,Faxis,squeeze(DAA(:,:,i_sensor)),'edgecolor','none');title(sensor_names2(i_sensor)) 
xline([1,3.5],'white','linewidth',1),xlabel('sec'),ylabel('Hz'),clim([-limit,limit]),axis tight; view(0,90); end,colormap redgreencmap

figure(4),clf
for i_sensor=65:128, subplot(8,8,i_sensor-64)
surf(time,Faxis,squeeze(DAA(:,:,i_sensor)),'edgecolor','none');title(sensor_names2(i_sensor)) 
xline([1,3.5],'white','linewidth',1),xlabel('sec'),ylabel('Hz'),clim([-limit,limit]),axis tight; view(0,90); end,colormap redgreencmap

%% Step5  
figure(5),clf
Sensor_Score=mean(squeeze(var(DAA(:,tstart+32:tend,:),[],2))); % a small offset in time 
%Sensor_Score=squeeze((mean(mean(abs(DAA(:,tstart:tend,:)),1),2)));
subplot(2,1,1),stem(Sensor_Score),xlabel('sensor #'),ylabel('Activation-score')
%threshold=quantile(Sensor_Score,.78);selected_sensor=find(Sensor_Score>threshold)
[~,list]=sort(Sensor_Score,'descend');slist=list(1:35)
subplot(2,1,2), plot(xyz(:,1),xyz(:,2),'ko',xyz(slist,1),xyz(slist,2),'r*')



