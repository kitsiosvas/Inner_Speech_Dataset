load subject1   % Y= Y: class | session#
STs=permute(X,[2,3,1]);clear X; STs_baseline=permute(baseline,[2,3,1]); clear baseline
[Nsensors,Ntime,Ntrials]=size(STs); Fs=double(fs); time=[1:Ntime]*(1/Fs);
class_labels=Y(:,1)+1; % Class 0-->1 "shift one" upwards
session_labels=Y(:,2); clear Y

%% average re-ref
%re_STs=[];for i_trial=1:Ntrials, ST_DATA=STs(:,:,i_trial); re_STs(:,:,i_trial)=ST_DATA-mean(ST_DATA);end
%STs=re_STs;

%% band-limiting brain-activity
%[b,a]=butter(3,[4 45]/(Fs/2));pSTs=permute(STs,[2,1,3]);filtered_STs=permute(filtfilt(b,a,pSTs),[2 1 3]);
%STs=filtered_STs;

% Calculate average for each class
AVE(:,:,1)=mean(STs(:,:,class_labels==1),3);AVE(:,:,2)=mean(STs(:,:,class_labels==2),3);
AVE(:,:,3)=mean(STs(:,:,class_labels==3),3);AVE(:,:,4)=mean(STs(:,:,class_labels==4),3);

figure(10),clf;
for ii=1:4  % For each class, visualize the average matrix
    subplot(2,2,ii);
    imagesc(AVE(:,:,ii));  % Visualize the 2D [128x1153] matrix
    clim([-max(abs(AVE),[],'all') max(abs(AVE),[],'all')]);  % Set color map limits to relative max and min values
    title(strcat('class-',num2str(ii)));
end
colormap redgreencmap



% AAA1=STs(:,:,class_labels==1); AA1=reshape(AAA1,[Nsensors*Ntime,size(AAA1,3)])';        %reshape(mean(AA1,2),[128,Ntime]) 
% AAA2=STs(:,:,class_labels==2); AA2=reshape(AAA2,[Nsensors*Ntime,size(AAA2,3)])'; 
% labels= [class_labels(class_labels==1);class_labels(class_labels==2)]
%  [~, Z] = rankfeatures([AA1;AA2]',labels,'criterion','ttest');
%          plot(time,reshape(Z,Nsensors,Ntime))

%% pairwise-computation of discriminability maps based on temporal-patterning   
DiscrMaps=[];
% Calculate discriminability maps (using Wilcoxon rank sum test) for each pair of classes
pair_no=0;
for i1=1:3
    for i2=i1+1:4
        pair_no = pair_no+1;
        % Reshape the 3D [128x1153x50] vector to 2D [128*1153, 50]. 
        % Also get the transpose --> [50, 128*1153] to stack both of the matrices together across trials
        AAA1 = STs(:,:,class_labels==i1); AA1 = reshape(AAA1, [Nsensors*Ntime, size(AAA1,3)])';
        AAA2 = STs(:,:,class_labels==i2); AA2 = reshape(AAA2, [Nsensors*Ntime, size(AAA2,3)])';
        paired_labels = [class_labels(class_labels==i1); class_labels(class_labels==i2)];
        
        % Run the Wilcoxon test. First argument is [147584x100] and second is [100x1].
        % Z is the absolute value of the critirion used [147584x1]
        [~, Z] = rankfeatures([AA1;AA2]', paired_labels, 'criterion', 'wilcoxon');

        % Reshape Z from [147584x1] to [128x1153] and save it to DiscrMaps. Ultimately we will have 6 of
        % those matrices-maps, one for each pair.
        DiscrMaps(:,:,pair_no) = reshape(Z, Nsensors, Ntime);
    end
end 

%% presenting & aggregating results & integrating in time to derive a sensor-specific score

% Visualize the 6 pairwise discriminability maps
figure(1), clf;
pair_no=0;
for i1=1:3
    for i2=i1+1:4
        pair_no=pair_no+1;
        subplot(2,3,pair_no)
        imagesc(DiscrMaps(:,:,pair_no))
        clim([0 max(DiscrMaps(:))]);
        title(strcat(num2str(i1),'-vs-',num2str(i2)));
    end
end
xlabel('sample #');
ylabel('sensor #');
colorbar, colormap hot;


% Visualise Avg and Max from the 6 discriminability maps
tstart=knnsearch(time',1); tend=knnsearch(time', 3.5); % action interval
figure(2),clf;

AVEmap = mean(DiscrMaps,3);
subplot(3,1,1);
imagesc(AVEmap);
ylabel('sensor #');
xlabel('sample #');
colorbar;
title('average across pairwise discriminability');
colormap hot;
xline([tstart tend], 'white', 'linewidth', 2)

MAXmap = max(DiscrMaps,[],3);
subplot(3,1,2);
imagesc(MAXmap);
ylabel('sensor #');
xlabel('sample #');
colorbar;
title('maximal pariwise diff');
colormap hot
xline([tstart tend], 'white', 'linewidth', 2)

% Plot the Sensor Score (from the AVG map matrix). For each of the 128 sensors, get the average across
% samples. Then keep the top 20% of those and plot their position
SensorScore=mean(AVEmap(:,tstart:tend),2);  %SensorScore2=max(AVEmap(:,tstart:end),[],2); %
subplot(3,1,3), plot(SensorScore),xlabel('sensor#')
[~,imax]=max(SensorScore);
threshold=quantile(SensorScore, .80);
selected_sensor=find(SensorScore>threshold);
load sensor_xyz.mat
figure(3), clf;
% Plot all the sensors and with red plot the selected ones.
plot(xyz(:,1), xyz(:,2), 'ko', xyz(selected_sensor,1), xyz(selected_sensor, 2), 'r*');

