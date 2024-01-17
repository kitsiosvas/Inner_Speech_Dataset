load subject1   % Y= Y: class | session#
STs=X;clear X;
%STs=permute(X,[2,3,1]);clear X; STs_baseline=permute(baseline,[2,3,1]); clear baseline
[Ntrials,Nsensors,Ntime]=size(STs); Fs=double(fs); time=[1:Ntime]*(1/Fs)
class_labels=double(Y(:,1))+1 % Class 0-->1 "shift one" upwards
session_labels=double(Y(:,2)); clear Y
load sensor_xyz

%average re-ref
%re_STs=[];for i_trial=1:Ntrials, ST_DATA=STs(:,:,i_trial); re_STs(:,:,i_trial)=ST_DATA-mean(ST_DATA);end
%STs=re_STs;
%re_STs_baseline=[];for i_trial=1:3, ST_DATA=STs_baseline(:,:,i_trial); re_STs_baseline(:,:,i_trial)=ST_DATA-mean(ST_DATA);end
%STs_baseline=re_STs_baseline;


%% band-limited filtering
[b,a]=butter(3,[13 30]/(Fs/2));
 f_STs=[]; for i_trial=1:Ntrials, STDATA=squeeze(STs(i_trial,:,:)); f_STs(i_trial,:,:)=filtfilt(b,a,STDATA')'; end
STs=f_STs;

%%
% Task
tpre=knnsearch(time',0.5); tstart=knnsearch(time',1);tend=knnsearch(time',3.5); 


trialCOV=[];for i_trial=1:Ntrials, E=squeeze(STs(i_trial,:,tstart:tend)); EE = E * E';
     covi=(EE./(trace(EE)))+ eye(Nsensors,Nsensors)*0.0001;
             trialCOV(:,:,i_trial)=covi; end
 
DD=[];for i1=1:Ntrials-1, for i2=i1+1:Ntrials
      DD(i1,i2)= distance_riemann(trialCOV(:,:,i1),trialCOV(:,:,i2));
      DD(i2,i1)=DD(i1,i2);end,end
M=cmdscale(DD,2);%plot(M(1:Nleft,1),M(1:Nleft,2),'bo',M(Nleft+1:end,1),M(Nleft+1:end,2),'r*')  
% figure(10),clf,subplot(2,1,1),plot(M(1:N1,1),M(1:N1,2),'bo',M(N1+1:end,1),M(N1+1:end,2),'r*'),axis equal
%                 legend('MI-left','MI-right')
% % [W,Pvalue,R] = WW_test(M(1:N1,:),M(N1+1:end,:));title(strcat('WWtest',':' ,num2str(W),' |  ',num2str(Pvalue)))

figure(1),clf,subplot(1,2,1),gscatter(M(:,1),M(:,2),session_labels,'k','sod'),hold on
gscatter(M(:,1),M(:,2),class_labels,'rgbk'),axis square,title("Single-trial Cov patterns - Session Shaped")
%plot(M(find(memi_labels==1),1),M(find(memi_labels==1),2),'.')


%%%%%% "Whitening" per Run based on KarcherMean("unsupervised learning step")
 CC=unique(session_labels);
RefCOVs_BOTH=[];for i=1:numel(CC),
  session_list=find(session_labels==CC(i)); KarcherMean=riemann_mean(trialCOV(:,:, session_list));
  RefCOVs(:,:,i)=KarcherMean;end

trialCOV2=[];for i_trial=1:Ntrials
   session_id=session_labels(i_trial);
   R=real(RefCOVs(:,:,session_id)); Ra=inv(sqrtm(R));
   trialCOV2(:,:,i_trial)= Ra*trialCOV(:,:,i_trial)*Ra;  %R^(-1/2)*trialCOV(:,:,i_trial)*R^(-1/2); % inv(sqrtm(align_matrix))
end

DD2=[];for i1=1:Ntrials-1, for i2=i1+1:Ntrials
      DD2(i1,i2)= distance_riemann(trialCOV2(:,:,i1),trialCOV2(:,:,i2));
      DD2(i2,i1)=DD2(i1,i2);end,end
M2=cmdscale(real(DD2),2);
%subplot(1,2,2),gscatter(M2(:,1),M2(:,2),class_labels,'k','sodh'),hold on
%gscatter(M2(:,1),M2(:,2),session_labels,'brg'),axis square,title("after Whitening")
subplot(1,2,2),gscatter(M2(:,1),M2(:,2),session_labels,'k','sod'),hold on
gscatter(M2(:,1),M2(:,2),class_labels,'rgbk'),axis square,title("Single-trial Cov patterns - Session shaped")






M=cmdscale(DD(81:160,81:160),2);
figure(1),clf,subplot(1,2,1),gscatter(M(:,1),M(:,2),class_labels(81:160),'k','sodh'),hold on



























% 
% 
% 
% %% Step1 : constructing wavelet filterbanks
% %[244 x 520 x 116]
% fb = cwtfilterbank('SignalLength',Ntime,'SamplingFrequency',Fs,'FrequencyLimits',[4 40],'VoicesPerOctave',10);
% 
% %% Step2  : Estimating  single-trial (time-varying) CWT-transform & computing averaged Scalograms profiles
% r=randperm(Ntrials); r1=r(1:round(Ntrials/2)); r2=setdiff([1:Ntrials],r1)
% STs_A=STs(:,:,r1); class_labels_A=class_labels(r1); % training set
% STs_B=STs(:,:,r2); class_labels_B=class_labels(r2); % hold out-out set
% 
% Ntrials_A=size(STs_A,3)
% DiscrMAPS=[]; WT_ALL=[];for i_sensor=1:Nsensors, i_sensor/Nsensors
% WT=[];for i_trial=1:Ntrials_A,
%      signal=STs_A(i_sensor,:,i_trial); [wt,Faxis,coi]=cwt(signal,'FilterBank',fb);  % knnsearch(Faxis,35)   knnsearch(Faxis,5)
%      wt=wt(:,tstart:tend); 
%      WT(:,:,i_trial)=abs(wt);end,
% AA=reshape(WT,[numel(Faxis)*size(wt,2),size(WT,3)])';   
% [idx,Z]=fscmrmr(AA,class_labels_A); Z(idx(11:end))=0;
% sensorDiscrMaps=reshape(Z,numel(Faxis),size(wt,2)); % locations in time-frequency plane of the 10 most discriminant moduli
% DiscrMAPS(:,:,i_sensor)=sensorDiscrMaps;
% WT_ALL(i_sensor,:,:,:)=WT;
% end
% 
% 
% [Nsensors ,Nfreq,Nsamples,Ntrial] =size(WT_ALL)
% ALL_features=[];ALL_rr=[]; ALL_cc=[]; for i_sensor=1:Nsensors
%  [rr,cc]=find(DiscrMAPS(:,:,i_sensor));
%  features=[];for i_ind=1:10,
%  features=[features,squeeze(WT_ALL(i_sensor,rr(i_ind),cc(i_ind),:))];end
%  ALL_features=[ALL_features,features];
%  ALL_rr=[ALL_rr,rr']; ALL_cc=[ALL_cc,cc']; 
%  end 
%  [idx,Z]=fscmrmr(ALL_features,class_labels_A);
% 
%  DData=[ALL_features(:,idx(1:100)),double(class_labels_A)];
%  % --> classification Learner app
%  %DData=[ALL_features(:,:),double(class_labels_A)]
%  %[trainedClassifier, validationAccuracy] = trainClassifier(DData)
% 
% %% Test Part
% Ntrials_B=size(STs_B,3)
% 
%  ALL_features_B=[]; for i_sensor=1:Nsensors, i_sensor/Nsensors
%    WT=[];for i_trial=1:Ntrials_B,
%      signal=STs_B(i_sensor,:,i_trial); [wt,Faxis,coi]=cwt(signal,'FilterBank',fb);  % knnsearch(Faxis,35)   knnsearch(Faxis,5)
%      wt=wt(:,tstart:tend); 
%      WT(:,:,i_trial)=abs(wt);end,
% [rr,cc]=find(DiscrMAPS(:,:,i_sensor));
%  features=[];for i_ind=1:10,
%  features=[features,squeeze(WT(rr(i_ind),cc(i_ind),:))];end
%  ALL_features_B=[ALL_features_B,features]; %ALL_rr=[ALL_rr,rr']; ALL_cc=[ALL_cc,cc']; 
%  end 
% 
%  DData_B=[ALL_features_B(:,idx(1:100)),double(class_labels_B)];
% 
%   [yfit,scores] = trainedClassifier.predictFcn(DData_B(:,1:end-1)) 
% Acc=sum(abs(yfit-double(class_labels_B))==0)/numel(class_labels_B)
% 
% 
% 
% % 
% % %% Step3 Presenting ta averaged Discriminant scalograms per sensor  (all pairs taken together) 
% % 
% % ALL_DiscrMAPS=DiscrMAPS;     %
% % uplimit=max(ALL_DiscrMAPS(:));lolimit=min(ALL_DiscrMAPS(:));
% % figure(1),clf
% % for i_sensor=1:64, subplot(8,8,i_sensor)
% % plot(Faxis,max(ALL_DiscrMAPS(:,:,i_sensor),[],2)),xlim([3,41]),ylim([0 uplimit]), title(sensor_names2(i_sensor)),
% % xlabel('Hz'),grid, end
% % 
% % 
% % figure(2),clf
% % for i_sensor=65:128, subplot(8,8,i_sensor-64)
% % plot(Faxis,max(ALL_DiscrMAPS(:,:,i_sensor),[],2)),xlim([3,41]),ylim([0 uplimit]) ,title(sensor_names2(i_sensor)),
% % xlabel('Hz'),grid, end
% % 
% % 
% % %% Step4 presenting topographically the most informative sensors 
% % %a. averaged score within the action-interval 
% % figure(3),clf
% % Sensor_Score=[];for i_sensor=1:Nsensors,Sensor_Score(i_sensor)=mean(mean(ALL_DiscrMAPS(:,tstart:tend,i_sensor)));end
% % subplot(2,1,1),stem(Sensor_Score),xlabel('sensor #'),ylabel('Activation-score')
% % %threshold=quantile(Sensor_Score,.78);selected_sensor=find(Sensor_Score>threshold)
% % [~,list]=sort(Sensor_Score,'descend');slist=list(1:15) % 25 most discriminative sensors --> a design- parameter
% % subplot(2,1,2), plot(xyz(:,1),xyz(:,2),'ko'),hold on,plot(xyz(slist,1),xyz(slist,2),'r.','markersize',15)
% % 
% % %b. emphasizing the highest values within the action-interval 
% % figure(4),clf
% % Sensor_Score=[];for i_sensor=1:Nsensors,Q=ALL_DiscrMAPS(:,tstart:tend,i_sensor);
% %                     Sensor_Score(i_sensor)=quantile(Q(:),0.95);end
% % subplot(2,1,1),stem(Sensor_Score),xlabel('sensor #'),ylabel('Activation-score')
% % %threshold=quantile(Sensor_Score,.78);selected_sensor=find(Sensor_Score>threshold)
% % [~,list]=sort(Sensor_Score,'descend');slist2=list(1:25) % 25 most discriminative sensors  --> a design- parameter
% % subplot(2,1,2), plot(xyz(:,1),xyz(:,2),'ko'),hold on,plot(xyz(slist2,1),xyz(slist2,2),'r.','markersize',15)
% % 
% % 
% % 
% 
% 
% 
% 
% 
% 
% 
