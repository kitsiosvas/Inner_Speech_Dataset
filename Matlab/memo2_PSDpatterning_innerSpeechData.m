load subject1   % Y= Y: class | session#
STs = permute(X,[2,3,1]); clear X; STs_baseline = permute(baseline,[2,3,1]); clear baseline
[Nsensors,Ntime,Ntrials] = size(STs); Fs=double(fs); time=[1:Ntime]*(1/Fs);
class_labels = Y(:,1)+1; % Class 0-->1 "shift one" upwards
session_labels = Y(:,2); clear Y
load sensor_xyz

%% average re-ref
%re_STs=[];for i_trial=1:Ntrials, ST_DATA=STs(:,:,i_trial); re_STs(:,:,i_trial)=ST_DATA-mean(ST_DATA);end
%STs=re_STs;

%% Valculate PSDs
% Task
tstart = knnsearch(time', 1); tend = knnsearch(time', 3.5); 
trialPSD = [];
for i_trial=1:Ntrials
    ST_DATA = STs(:,tstart:tend,i_trial); 
    [STpsd, faxis] = pspectrum(ST_DATA', Fs, 'FrequencyLimits', [1 45], 'FrequencyResolution', 2);
    trialPSD(:,:,i_trial) = STpsd';
end 
% Baseline 
restPSD=[];
for i_trial=1:3
    ST_DATA = STs_baseline(:,:,i_trial); 
    [STpsd, faxis2] = pspectrum(ST_DATA', Fs, 'FrequencyLimits', [1 45], 'FrequencyResolution', 2);
    restPSD(:,:,i_trial) = STpsd';
end 

% Calculate average PSD for each class + baseline
AVE_PSDrest=mean(restPSD,3);
AVE_PSD(:,:,1)=trimmean(trialPSD(:,:,class_labels==1),10,'round',3);AVE_PSD(:,:,2)=trimmean(trialPSD(:,:,class_labels==2),10,'round',3);
AVE_PSD(:,:,3)=trimmean(trialPSD(:,:,class_labels==3),10,'round',3);AVE_PSD(:,:,4)=trimmean(trialPSD(:,:,class_labels==4),10,'round',3);

% figure(1),clf, subplot(3,2,1),imagesc(pow2db(AVE_PSDrest)),caxis([pow2db(min(AVE_PSDrest(:))) pow2db(max(AVE_PSDrest(:)))])
%                subplot(3,2,3),imagesc(pow2db(AVE_PSD(:,:,1))),  caxis([pow2db(min(AVE_PSDrest(:))) pow2db(max(AVE_PSDrest(:)))])
%                subplot(3,2,4),imagesc(pow2db(AVE_PSD(:,:,2))), caxis([pow2db(min(AVE_PSDrest(:))) pow2db(max(AVE_PSDrest(:)))])
%                 subplot(3,2,5),imagesc(pow2db(AVE_PSD(:,:,3))), caxis([pow2db(min(AVE_PSDrest(:))) pow2db(max(AVE_PSDrest(:)))])
%                 subplot(3,2,6),imagesc(pow2db(AVE_PSD(:,:,4))), caxis([pow2db(min(AVE_PSDrest(:))) pow2db(max(AVE_PSDrest(:)))])

% Plot the 5 AVG PSDs
figure(1),clf; 
subplot(3,2,1), plot(faxis, pow2db(AVE_PSDrest))   , title('baseline'), grid,ylim([-130 -90])
subplot(3,2,3), plot(faxis, pow2db(AVE_PSD(:,:,1))), title('class1')  , grid,ylim([-130 -90])
subplot(3,2,4), plot(faxis, pow2db(AVE_PSD(:,:,2))), title('class2')  , grid,ylim([-130 -90])
subplot(3,2,5), plot(faxis, pow2db(AVE_PSD(:,:,3))), title('class3')  , grid,ylim([-130 -90])
subplot(3,2,6), plot(faxis, pow2db(AVE_PSD(:,:,4))), title('class4')  , grid,ylim([-130 -90])
              
% Plot the 4 AVG PSDs in % form after substracting the baseline AVG 
figure(2),clf;
for ii=1:4
    subplot(2,2,ii), plot(faxis, 100*(AVE_PSD(:,:,ii)-AVE_PSDrest)./AVE_PSDrest);
    grid;
    title(strcat('Relative Change-',num2str(ii)));
    xlabel('Hz'), ylabel('%')
end

% Same but use image plot for resulted matrices
figure(3),clf;
for ii=1:4
    subplot(2,2,ii);
    imagesc((AVE_PSD(:,:,ii)-AVE_PSDrest)./AVE_PSDrest);
    clim([-1 1]);
    title(strcat('Relative Change-',num2str(ii)));
    q = xticks; 
    xticklabels(round(faxis(q)));
    xlabel('Hz'), ylabel('sensor #'); 
    colorbar
end
colormap redgreencmap


% Get the mean of all 4 classes and plot it
figure(4), clf;
subplot(2,2,1);
imagesc((mean(AVE_PSD,3)-AVE_PSDrest)./AVE_PSDrest);
clim([-1 1]);
q = xticks;
xticklabels(round(faxis(q)));
xlabel('Hz'), ylabel('sensor #');
colorbar;
colormap redgreencmap;

% Again the AVG of all classes with line plot as initially
subplot(2,2,2),plot(faxis,(mean(AVE_PSD,3)-AVE_PSDrest)./AVE_PSDrest),xlabel('Hz')

% Get the Sensor Score
SS = (mean(AVE_PSD,3)-AVE_PSDrest)./AVE_PSDrest; 
Sensor_Score = (mean(SS>0.1, 2));
subplot(2,2,3);
stem(Sensor_Score);
xlabel('sensor #'), ylabel('ERSscore');
threshold = quantile(Sensor_Score, .75);
selected_sensor = find(Sensor_Score>threshold);
subplot(2,2,4);
plot(xyz(:,1), xyz(:,2), 'ko', xyz(selected_sensor,1), xyz(selected_sensor,2), 'r*')


%%  PSD-pattern Discriminability 
[Nsensors,Nfrequencies,Ntrials]=size(trialPSD)

% pairwise-computation of discriminability maps based on PSD-patterning   
DiscrMaps=[];pair_no=0;
for i1=1:3
    for i2=i1+1:4
        pair_no=pair_no+1;
        AAA1=trialPSD(:,:,class_labels==i1); AA1=reshape(AAA1,[Nsensors*Nfrequencies,size(AAA1,3)])';         
        AAA2=trialPSD(:,:,class_labels==i2); AA2=reshape(AAA2,[Nsensors*Nfrequencies,size(AAA2,3)])';
        paired_labels = [class_labels(class_labels==i1);
        class_labels(class_labels==i2)];
        [~, Z] = rankfeatures([AA1;AA2]', paired_labels, 'criterion', 'ttest');
        DiscrMaps(:,:,pair_no) = reshape(Z,Nsensors,Nfrequencies);
    end
end 

%% presenting & aggregating results & integrating in time to derive a sensor- specific score
figure(1), clf;
pair_no = 0;
for i1=1:3
    for i2=i1+1:4
        pair_no=pair_no+1;
        subplot(2,3,pair_no),
        imagesc(DiscrMaps(:,:,pair_no)),
        clim([0 max(DiscrMaps(:))]),
        title(strcat(num2str(i1),'-vs-',num2str(i2))), 
        q=xticks;
        xticklabels(round(faxis(q)));
        colorbar
        xlabel('Hz'),ylabel('sensor #');
    end
end
colormap hot

% Plot AVG and MAX discriminability maps again
figure(2), clf,
AVEmap = mean(DiscrMaps,3); 
MAXmap=max(DiscrMaps,[],3);

subplot(3,1,1);
imagesc(AVEmap);
q=xticks;
xticklabels(round(faxis(q)));
xlabel('Hz'), ylabel('sensor #');
colorbar
title('average across pairwise discriminability')
colormap hot

subplot(3,1,2)
imagesc(MAXmap)
colorbar
title('maximal pariwise diff')
colormap hot

%  Plot the Sensor Score (from the AVG map matrix). For each of the 128 sensors, get the average across
% samples
SensorScore = mean(AVEmap,2);  
subplot(3,1,3);
plot(SensorScore);
xlabel('sensor#');
[~, imax] = max(SensorScore); % position of the max valued electrode
threshold = quantile(SensorScore, .80);
selected_sensor = find(SensorScore>threshold);

figure(3),clf, plot(xyz(:,1),xyz(:,2),'ko',xyz(selected_sensor,1),xyz(selected_sensor,2),'r*',xyz(imax,1),xyz(imax,2),'g*')

