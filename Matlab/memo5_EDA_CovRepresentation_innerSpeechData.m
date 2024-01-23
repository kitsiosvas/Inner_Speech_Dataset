load subject1   % Y= Y: class | session#
STs=X;clear X;
%STs=permute(X,[2,3,1]);clear X; STs_baseline=permute(baseline,[2,3,1]); clear baseline
[Ntrials,Nsensors,Ntime]=size(STs); Fs=double(fs); time=[1:Ntime]*(1/Fs);
class_labels=double(Y(:,1))+1; % Class 0-->1 "shift one" upwards
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








