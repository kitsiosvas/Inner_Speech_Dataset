topPicksTtest = load("topPicksTtest.mat");
topPicksTtest = topPicksTtest.topPicks;


topPicksWilcoxon = load("topPicksWilcoxon.mat");
topPicksWilcoxon = topPicksWilcoxon.topPicks;



picks = topPicksWilcoxon;

% Find unique numbers and their counts
uniqueNumbers = unique(picks);
counts = histc(picks(:), uniqueNumbers);

[counts, idx] = sort(counts, 'descend');
uniqueNumbers = uniqueNumbers(idx);

% Display the results
disp('Number   Frequency');
disp('-----------------');
for i = 1:length(uniqueNumbers)
    fprintf('%4d     %8d\n', uniqueNumbers(i), counts(i));
end



% Create a scatter plot with marker size based on counts
figure;
h = scatter(xyz(uniqueNumbers, 1), xyz(uniqueNumbers, 2), 25 * counts, counts, 'filled');
row1 = dataTipTextRow('Sensor#', uniqueNumbers);
row2 = dataTipTextRow('Name', sensor_names(uniqueNumbers));
row3 = dataTipTextRow('Count', counts);
h.DataTipTemplate.DataTipRows = [row1; row2; row3];

colormap(jet); % You can choose a different colormap if needed
colorbar;
title('Most informative sensors voting scheme');


% Use only integer values on the colorbar
clim([1 6]); % Set the color limits
ticks = 1:6; % Set the desired tick locations
cbar = colorbar;
cbar.Ticks = ticks;
