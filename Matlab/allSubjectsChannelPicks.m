s1 = [11, 12, 14, 13, 10, 15, 16, 8, 9, 71, 126, 25, 24, 17, 18, 128, 7, 127, 5, 108, 116, 115, ...
      122, 107, 119, 45, 6, 114, 113, 100, 112, 118, 125, 37, 38];

s2 = [13, 82, 92, 77, 93, 70, 94, 71, 84, 40, 12, 91, 74, 89, 53, 51, 41, 76, 10, 73, 11, 83, ...
      50, 66, 99, 43, 90, 96, 14, 97, 49, 95, 101, 35, 45];

s3 = [58, 57, 60, 114, 74, 115, 71, 116, 56, 7, 59, 5, 20, 80, 111, 61, 6, 113, 67, 21, 123, 110, ...
      126, 125, 24, 76, 68, 41, 124, 77, 17, 106, 19, 2, 27];

s4 = [79, 93, 78, 80, 20, 42, 39, 41, 44, 4, 19, 24, 40, 21, 103, 102, 45, 38, 5, 22, 25, 43, ...
      82, 92, 71, 18, 37, 70, 29, 28, 57, 69, 23, 113, 85];

s5 = [92, 93, 79, 77, 82, 81, 78, 76, 83, 80, 96, 95, 84, 85, 89, 91, 73, 90, 74, 99, 88, 116, ...
      100, 101, 115, 30, 38, 28, 29, 37, 94, 39, 27, 67, 45];

s6 = [90, 127, 79, 78, 3, 2, 74, 83, 10, 11, 57, 87, 68, 14, 101, 126, 53, 128, 1, 125, 66, 52, ...
      4, 94, 89, 86, 48, 17, 65, 51, 120, 100, 115, 76, 41];

s7 = [9, 8, 57, 125, 54, 53, 121, 10, 72, 42, 7, 74, 16, 123, 65, 100, 11, 55, 73, 75, 110, 17, ...
      122, 25, 33, 64, 52, 6, 109, 115, 18, 107, 113, 47, 124];

s8 = [25, 44, 121, 28, 122, 120, 124, 8, 23, 24, 9, 37, 38, 126, 41, 40, 39, 123, 125, 128, 21, 10, ...
      7, 19, 4, 22, 127, 6, 15, 12, 117, 11, 72, 114, 116];

s9 = [124, 113, 121, 120, 114, 6, 123, 116, 115, 112, 125, 122, 7, 16, 35, 110, 5, 51, 19, 4, 49, 128, ...
      50, 8, 18, 34, 48, 17, 55, 36, 53, 45, 54, 21, 109];

s10 = [29, 16, 38, 37, 48, 58, 30, 17, 39, 45, 7, 44, 46, 106, 18, 22, 28, 21, 23, 36, 119, 122, ...
       118, 120, 31, 105, 24, 113, 123, 43, 25, 42, 121, 124, 126];


% Combine all lists into a single matrix
allLists = [s1; s2; s3; s4; s5; s6; s7; s8; s9; s10];

% Find unique numbers and their counts
uniqueNumbers = unique(allLists);
counts = histc(allLists(:), uniqueNumbers);

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
