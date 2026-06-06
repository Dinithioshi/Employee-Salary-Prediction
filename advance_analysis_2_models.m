clc; clear;
rng(5677432);
data = readtable('Employee data.csv'); %read table
%% 
sum(ismissing(data)) %identify missing values
%% 
%identify missing values by variable
fprintf('Missing Values by Variable:\n');

% id
fprintf('id: %d\n', sum(ismissing(data.id)));

% gender
fprintf('gender: %d\n', sum(ismissing(data.gender)));

% bdate
fprintf('bdate: %d\n', sum(ismissing(data.bdate)));

% educ (0 = missing)
fprintf('educ: %d\n', sum(data.educ == 0));

% jobcat (0 = missing)
fprintf('jobcat: %d\n', sum(data.jobcat == 0));

% salary (0 = missing)
fprintf('salary: %d\n', sum(data.salary == 0));

% salbegin (0 = missing)
fprintf('salbegin: %d\n', sum(data.salbegin == 0));

% jobtime (0 = missing)
fprintf('jobtime: %d\n', sum(data.jobtime == 0));

% prevexp (0 = missing)
fprintf('prevexp: %d\n', sum(data.prevexp == 0));

% minority (9 = missing)
fprintf('minority: %d\n', sum(data.minority == 9));
%% 
sum(isnan(data.prevexp))
%% 
data.prevexp(data.prevexp == 0) = NaN;

%% 
data = rmmissing(data);%remove missing values
%% 
%divide train test
rng(1);
cv = cvpartition(height(data),'HoldOut',0.2);  % 20% test

train_data = data(training(cv), :);  % 80% train
test_data  = data(test(cv), :);      % 20% test

fprintf('Training set size: %d\n', height(train_data));
fprintf('Testing set size: %d\n', height(test_data));



%% 

%create age
class(train_data.bdate)
class(test_data.bdate)
%% 
% Create Age in completed years
% Assume analysis year is 1995
fixed_today = datetime(1995,1,1);  % January 1, 1995

train_data.Age = floor(years(fixed_today - train_data.bdate));
test_data.Age = floor(years(fixed_today - test_data.bdate));
%% 

% Quick check
summary(train_data(:,{'Age'}))   % works on table column
summary(test_data(:,{'Age'}))
%% 
%histogram of age
figure;
histogram(train_data.Age, 10)  % 10 bins, you can adjust
xlabel('Age (years)')
ylabel('Number of Employees')
title('Distribution of Employee Ages')
grid on
%% 
%After creating the Age variable, the original 
% identifiers (id and bdate) are no longer needed for analysis, so it’s safe to remove them.
train_data(:, {'id','bdate'}) = [];
test_data(:, {'id','bdate'}) = [];
%% 
%One-Hot Encode jobcat
jobcat_idx = train_data.jobcat';  % transpose because ind2vec expects row vector
jobcat_onehot = full(ind2vec(jobcat_idx));
jobcat_onehot = jobcat_onehot';  % now n × 3

%% 
%without scaaling 0/1
X2 = [train_data.Age, train_data.educ, train_data.salary, train_data.salbegin, train_data.jobtime, train_data.prevexp];
%% 
% Standardize all variables
[X_scaled1, mu, sigma]= zscore(X2);
X_scaled=[X_scaled1,train_data.minority, jobcat_onehot];
%% k=4
[idx4,cent4,sumdist] = kmeans(X_scaled,4,'dist','sqeuclidean',...
'display','final','replicates',20);
[silh4,h] = silhouette(X_scaled,idx4,'sqeuclidean');
xlabel('Silhouette Value');
ylabel('Cluster');
mean(silh4)
%% k=3
[idx3,cent3,sumdist] = kmeans(X_scaled,3,'dist','sqeuclidean',...
'display','final','replicates',20);
[silh3,h] = silhouette(X_scaled,idx3,'sqeuclidean');
xlabel('Silhouette Value')
ylabel('Cluster')
mean(silh3)
%% k=2
[idx2,cent2,sumdist] = kmeans(X_scaled,2,'dist','sqeuclidean',...
'display','final','replicates',20);
[silh2,h] = silhouette(X_scaled,idx2,'sqeuclidean');
xlabel('Silhouette Value');
ylabel('Cluster');
mean(silh2)
%% k=2 is best
%% 
%Add cluster labels to train set
train_data.Cluster = idx2;  % cluster assignment for k = 2
%%prepare test set to add cluster lables
% We must repeat the exact same steps on test_data

% 1. One-Hot Encode Test Jobcat
jobcat_idx_test = test_data.jobcat';
jobcat_onehot_test = full(ind2vec(jobcat_idx_test));
jobcat_onehot_test = jobcat_onehot_test'; 

% FIX: Ensure Test One-Hot has same number of columns as Train
% (If test data is missing "jobcat 3", this adds a column of zeros to match dimensions)
if size(jobcat_onehot_test, 2) < size(jobcat_onehot, 2)
    jobcat_onehot_test(:, size(jobcat_onehot, 2)) = 0; 
end

% 2. Create Numeric Matrix for Test
X2_test = [test_data.Age, test_data.educ, test_data.salary, ...
           test_data.salbegin, test_data.jobtime, test_data.prevexp];

% 3. Standardize Test Data
% CRITICAL: Use the 'mu' and 'sigma' from the TRAINING set
X_scaled1_test = (X2_test - mu) ./ sigma;

% 4. Combine to create final Test Matrix
test_scaled = [X_scaled1_test, test_data.minority, jobcat_onehot_test];

%% apply cluster lables to test set
% Calculate distance from every test point to the 2 centroids
distances = pdist2(test_scaled, cent2, 'squaredeuclidean');

% Find the nearest cluster (min distance)
[~, test_cluster_idx] = min(distances, [], 2);

% Add to test table
test_data.Cluster = test_cluster_idx;



%%
%summary of each cluster in train data
% Number of employees per cluster
tabulate(train_data.Cluster)

% Save the summary to a variable and display it
cluster_summary = grpstats(train_data, 'Cluster', {'mean','median','min','max'}, ...
         'DataVars', {'Age','educ','salary','salbegin','jobtime','prevexp','minority'});

% Display the table in the Command Window
disp(cluster_summary)
%%
%Split train set by gender 
% Male
male = train_data(strcmp(train_data.gender,'m'),:);

% Female
female = train_data(strcmp(train_data.gender,'f'),:);
%% 
%Male
X_male = [male.Age, male.educ, male.salbegin, male.jobtime, male.prevexp, male.minority, male.jobcat,male.Cluster];  
Y_male = male.salary;
% Fit tree (NO CrossVal here)
tree_male = fitrtree(X_male, Y_male,...
    'PredictorNames',{'Age','educ','salbegin','jobtime','prevexp','minority','jobcat','Cluster'},...
    'CategoricalPredictors',{'minority','jobcat','Cluster'});
% Cross-validated loss for all pruning levels
[E1, SE1, NLEAF1, BESTLEVEL1] = ...
    cvloss(tree_male,'SubTrees','all');
% Prune tree
tree_pruned = prune(tree_male,'Level',BESTLEVEL1);

% Visualize
view(tree_pruned,'Mode','graph');

% MSE on test data
male_test = test_data(strcmp(test_data.gender,'m'),: );
X_male_test = [male_test.Age, male_test.educ, male_test.salbegin, male_test.jobtime, male_test.prevexp, male_test.minority, male_test.jobcat,male_test.Cluster];  
Y_male_test = male_test.salary;
L1=loss(tree_pruned,X_male_test,Y_male_test );



%% 
%Female
X_female = [female.Age, female.educ, female.salbegin, female.jobtime, female.prevexp, female.minority, female.jobcat,female.Cluster];  
Y_female = female.salary;
% Fit tree (NO CrossVal here)
tree_female = fitrtree(X_female, Y_female,...
    'PredictorNames',{'Age','educ','salbegin','jobtime','prevexp','minority','jobcat','Cluster'},...
    'CategoricalPredictors',{'minority','jobcat','Cluster'});
% Cross-validated loss for all pruning levels
[E2, SE2, NLEAF2, BESTLEVEL2] = ...
    cvloss(tree_female,'SubTrees','all');
% Prune tree
tree_prunedf = prune(tree_female,'Level',BESTLEVEL2);

% Visualize
view(tree_prunedf,'Mode','graph');

% MSE on test data
female_test = test_data(strcmp(test_data.gender,'f'),:);
X_female_test = [female_test.Age, female_test.educ, female_test.salbegin, female_test.jobtime, female_test.prevexp, female_test.minority, female_test.jobcat,female_test.Cluster];  
Y_female_test = female_test.salary;
L2=loss(tree_prunedf,X_female_test,Y_female_test );


%% --- SAFETY CHECK: Print sizes of groups ---
fprintf('Male Count: %d\n', height(male));
fprintf('Female Count: %d\n', height(female));

% Warning if any group is too small
if height(male) < 30 || height(female) < 30
    warning('One of your groups is small (<30). Results may be less stable.');
else
    disp('Sample sizes are sufficient for regression trees.');
end

%% 
%% --- Separate Feature Importance Plots ---
names = tree_male.PredictorNames;

% 1. Plot for Male Tree
figure('Name', 'Male Feature Importance');
bar(predictorImportance(tree_pruned), 'FaceColor', [0.2 0.4 0.6]);
set(gca, 'XTick', 1:length(names), 'XTickLabel', names, 'XTickLabelRotation', 45);
ylabel('Importance Score');
title('Predictor Importance: Male Salary Model');
grid on;

% 2. Plot for Female Tree
figure('Name', 'Female Feature Importance');
bar(predictorImportance(tree_prunedf), 'FaceColor', [0.6 0.2 0.4]);
set(gca, 'XTick', 1:length(names), 'XTickLabel', names, 'XTickLabelRotation', 45);
ylabel('Importance Score');
title('Predictor Importance: Female Salary Model');
grid on;
%% 
%% --- Stacked Bar Chart: Cluster vs Job Category (Percentage) ---

% 1. Create contingency table: rows = Job Category, cols = Cluster
tbl = crosstab(train_data.jobcat, train_data.Cluster);

% 2. Transpose so Clusters are on the X-axis
tbl_transposed = tbl'; 

% 3. Convert raw counts to percentages
% We divide each row by its sum and multiply by 100
tbl_percent = (tbl_transposed ./ sum(tbl_transposed, 2)) * 100;

% 4. Create the stacked bar chart
figure('Name', 'Job Category Percentage per Cluster');
bar(tbl_percent, 'stacked');

% 5. Format the chart labels
xlabel('Cluster Number');
ylabel('Percentage (%)'); % Updated to show Percentage
title('Job Category Composition per Cluster (Percentage)');
ylim([0 100]); % Force Y-axis to 100%

% 6. Create specific Legend using your category names
job_names = {'Clerical', 'Custodial', 'Manager'};
legend(job_names, 'Location', 'northeastoutside');

% 7. Clean up the X-axis for the Clusters
xticks(1:size(tbl_percent, 1));
xticklabels({'Cluster 1', 'Cluster 2'});
grid on;
%% 
%% --- Cluster Profiling: Boxplots (Increased Font Size) ---
figure('Name', 'Numeric Distributions by Cluster');
fs = 14; % Define a variable for font size for easy adjustments

% Salary Comparison
subplot(1,3,1);
boxplot(train_data.salary, train_data.Cluster);
title('Salary by Cluster', 'FontSize', fs);
ylabel('Current Salary ($)', 'FontSize', fs);
xlabel('Cluster', 'FontSize', fs);
set(gca, 'FontSize', fs); % Sets the axis tick labels font size

% Education Comparison
subplot(1,3,2);
boxplot(train_data.educ, train_data.Cluster);
title('Education by Cluster', 'FontSize', fs);
ylabel('Years of Education', 'FontSize', fs);
xlabel('Cluster', 'FontSize', fs);
set(gca, 'FontSize', fs);

% Age Comparison
subplot(1,3,3);
boxplot(train_data.Age, train_data.Cluster);
title('Age by Cluster', 'FontSize', fs);
ylabel('Age (Years)', 'FontSize', fs);
xlabel('Cluster', 'FontSize', fs);
set(gca, 'FontSize', fs);
%% 
%% --- Cluster Profiling: Mean Comparison ---
% Calculate means for the numeric variables
stats = grpstats(train_data, 'Cluster', 'mean', 'DataVars', {'salary', 'salbegin', 'educ', 'Age', 'prevexp'});

% Plot the means
figure('Name', 'Average Characteristics per Cluster');
bar(stats{:, 3:end}'); % Excludes 'GroupCount' and 'Cluster' index
set(gca, 'XTickLabel', {'Salary', 'Start Salary', 'Educ', 'Age', 'Prev Exp'}, 'XTickLabelRotation', 45);
legend({'Cluster 1', 'Cluster 2'});
ylabel('Mean Value');
title('Average Employee Profile by Cluster');
grid on;
%% 
%% --- Fixed Cluster Profiling: Gender Composition ---

% 1. Create a table of counts
gcounts = groupsummary(train_data, {'Cluster', 'gender'});

% 2. Unstack to get genders as columns
% This ensures 'f' and 'm' are explicitly labeled columns
plotData = unstack(gcounts, 'GroupCount', 'gender');

% 3. Extract just the numeric data (Columns 2 and 3: f and m)
% We convert to percentage: (row / row_sum) * 100
countsMatrix = table2array(plotData(:, 2:end));
percentMatrix = (countsMatrix ./ sum(countsMatrix, 2)) * 100;

% 4. Plot
figure('Name', 'Corrected Gender Percentage');
b = bar(percentMatrix, 'stacked');

% 5. Formatting
xticklabels({'Cluster 1', 'Cluster 2'});
xlabel('Cluster Number');
ylabel('Percentage (%)');
title('Gender Composition (Corrected)');
legend(plotData.Properties.VariableNames(2:end), 'Location', 'northeastoutside');
grid on;