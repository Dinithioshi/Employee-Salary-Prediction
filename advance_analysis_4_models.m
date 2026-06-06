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
%Split train set by gender and cluster
% Male, Cluster 1
male_cluster1 = train_data(strcmp(train_data.gender,'m') & train_data.Cluster == 1, :);

% Female, Cluster 1
female_cluster1 = train_data(strcmp(train_data.gender,'f') & train_data.Cluster == 1, :);

% Male, Cluster 2
male_cluster2 = train_data(strcmp(train_data.gender,'m') & train_data.Cluster == 2, :);

% Female, Cluster 2
female_cluster2 = train_data(strcmp(train_data.gender,'f') & train_data.Cluster == 2, :);
%% 
%Male Cluster 1
X_male_c1 = [male_cluster1.Age, male_cluster1.educ, male_cluster1.salbegin, male_cluster1.jobtime, male_cluster1.prevexp, male_cluster1.minority, male_cluster1.jobcat];  
Y_male_c1 = male_cluster1.salary;
% Fit tree (NO CrossVal here)
tree_male_c1 = fitrtree(X_male_c1, Y_male_c1, ...
    'PredictorNames',{'Age','educ','salbegin','jobtime','prevexp','minority','jobcat'});

% Cross-validated loss for all pruning levels
[E1, SE1, NLEAF1, BESTLEVEL1] = ...
    cvloss(tree_male_c1,'SubTrees','all');
% Prune tree
tree_pruned = prune(tree_male_c1,'Level',BESTLEVEL1);

% Visualize
view(tree_pruned,'Mode','graph');

% MSE on test data
male_cluster1_test = test_data(strcmp(test_data.gender,'m') & test_data.Cluster == 1, :);
X_male_c1_test = [male_cluster1_test.Age, male_cluster1_test.educ, male_cluster1_test.salbegin, male_cluster1_test.jobtime, male_cluster1_test.prevexp, male_cluster1_test.minority, male_cluster1_test.jobcat];  
Y_male_c1_test = male_cluster1_test.salary;
L1=loss(tree_pruned,X_male_c1_test,Y_male_c1_test );


%% 
%Female Cluster 1
X_female_c1 = [female_cluster1.Age, female_cluster1.educ, female_cluster1.salbegin, female_cluster1.jobtime, female_cluster1.prevexp, female_cluster1.minority, female_cluster1.jobcat];  
Y_female_c1 = female_cluster1.salary;
% Fit tree (NO CrossVal here)
tree_female_c1 = fitrtree(X_female_c1, Y_female_c1, ...
    'PredictorNames',{'Age','educ','salbegin','jobtime','prevexp','minority','jobcat'});

% Cross-validated loss for all pruning levels
[E2, SE2, NLEAF2, BESTLEVEL2] = ...
    cvloss(tree_female_c1,'SubTrees','all');
% Prune tree
tree_pruned1f = prune(tree_female_c1,'Level',BESTLEVEL2);

% Visualize
view(tree_pruned1f,'Mode','graph');

% MSE on test data
female_cluster1_test = test_data(strcmp(test_data.gender,'f') & test_data.Cluster == 1, :);
X_female_c1_test = [female_cluster1_test.Age, female_cluster1_test.educ, female_cluster1_test.salbegin, female_cluster1_test.jobtime, female_cluster1_test.prevexp, female_cluster1_test.minority, female_cluster1_test.jobcat];  
Y_female_c1_test = female_cluster1_test.salary;
L2=loss(tree_pruned1f,X_female_c1_test,Y_female_c1_test );

%% 
%Male Cluster 2
X_male_c2 = [male_cluster2.Age, male_cluster2.educ, male_cluster2.salbegin, male_cluster2.jobtime, male_cluster2.prevexp, male_cluster2.minority, male_cluster2.jobcat];  
Y_male_c2 = male_cluster2.salary;
% Fit tree (NO CrossVal here)
tree_male_c2 = fitrtree(X_male_c2, Y_male_c2, ...
    'PredictorNames',{'Age','educ','salbegin','jobtime','prevexp','minority','jobcat'});

% Cross-validated loss for all pruning levels
[E3, SE3, NLEAF3, BESTLEVEL3] = ...
    cvloss(tree_male_c2,'SubTrees','all');
% Prune tree
tree_pruned3 = prune(tree_male_c2,'Level',BESTLEVEL3);

% Visualize
view(tree_pruned3,'Mode','graph');

% MSE on test data
male_cluster2_test = test_data(strcmp(test_data.gender,'m') & test_data.Cluster == 2, :);
X_male_c2_test = [male_cluster2_test.Age, male_cluster2_test.educ, male_cluster2_test.salbegin, male_cluster2_test.jobtime, male_cluster2_test.prevexp, male_cluster2_test.minority, male_cluster2_test.jobcat];  
Y_male_c2_test = male_cluster2_test.salary;
L3=loss(tree_pruned3,X_male_c2_test,Y_male_c2_test );
%% 
%Female Cluster 2
X_female_c2 = [female_cluster2.Age, female_cluster2.educ, female_cluster2.salbegin, female_cluster2.jobtime, female_cluster2.prevexp, female_cluster2.minority,female_cluster2.jobcat];  
Y_female_c2 = female_cluster2.salary;
% Fit tree (NO CrossVal here)
tree_female_c2 = fitrtree(X_female_c2, Y_female_c2, ...
    'PredictorNames',{'Age','educ','salbegin','jobtime','prevexp','minority','jobcat'});

% Cross-validated loss for all pruning levels
[E4, SE4, NLEAF4, BESTLEVEL4] = ...
    cvloss(tree_female_c2,'SubTrees','all');
% Prune tree
tree_pruned4 = prune(tree_female_c2,'Level',BESTLEVEL4);

% Visualize
view(tree_pruned4,'Mode','graph');

% MSE on test data
female_cluster2_test = test_data(strcmp(test_data.gender,'f') & test_data.Cluster == 2, :);
X_female_c2_test = [female_cluster2_test.Age, female_cluster2_test.educ, female_cluster2_test.salbegin, female_cluster2_test.jobtime, female_cluster2_test.prevexp, female_cluster2_test.minority, female_cluster2_test.jobcat];  
Y_female_c2_test = female_cluster2_test.salary;
L4=loss(tree_pruned4,X_female_c2_test,Y_female_c2_test );
%% 
% --- SAFETY CHECK: Print sizes of sub-groups ---
fprintf('Male Cluster 1 Count: %d\n', height(male_cluster1));
fprintf('Female Cluster 1 Count: %d\n', height(female_cluster1));
fprintf('Male Cluster 2 Count: %d\n', height(male_cluster2));
fprintf('Female Cluster 2 Count: %d\n', height(female_cluster2));

% Warning if any group is too small (e.g., less than 10 people)
if height(male_cluster1) < 10 || height(female_cluster1) < 10 || ...
   height(male_cluster2) < 10 || height(female_cluster2) < 10
    warning('One of your sub-groups is very small. The regression tree might fail or be inaccurate.');
end
%% 
%% --- Feature Importance: 4 Separate Plots ---

% Define common predictor names for all trees
predictor_names = {'Age','educ','salbegin','jobtime','prevexp','minority','jobcat'};

% 1. Male Cluster 1
figure('Name', 'Importance: Male Cluster 1');
imp1 = predictorImportance(tree_pruned); % your pruned male c1 tree
bar(imp1, 'FaceColor', [0.2 0.4 0.6]);
set(gca, 'XTick', 1:length(predictor_names), 'XTickLabel', predictor_names, 'XTickLabelRotation', 45);
title('Feature Importance: Male (Cluster 1)');
ylabel('Importance Score');
grid on;

% 2. Female Cluster 1
figure('Name', 'Importance: Female Cluster 1');
imp2 = predictorImportance(tree_pruned1f); % your pruned female c1 tree
bar(imp2, 'FaceColor', [0.6 0.2 0.4]);
set(gca, 'XTick', 1:length(predictor_names), 'XTickLabel', predictor_names, 'XTickLabelRotation', 45);
title('Feature Importance: Female (Cluster 1)');
ylabel('Importance Score');
grid on;

% 3. Male Cluster 2
figure('Name', 'Importance: Male Cluster 2');
imp3 = predictorImportance(tree_pruned3); % your pruned male c2 tree
bar(imp3, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTick', 1:length(predictor_names), 'XTickLabel', predictor_names, 'XTickLabelRotation', 45);
title('Feature Importance: Male (Cluster 2)');
ylabel('Importance Score');
grid on;

% 4. Female Cluster 2
figure('Name', 'Importance: Female Cluster 2');
imp4 = predictorImportance(tree_pruned4); % your pruned female c2 tree
bar(imp4, 'FaceColor', [0.8 0.2 0.6]);
set(gca, 'XTick', 1:length(predictor_names), 'XTickLabel', predictor_names, 'XTickLabelRotation', 45);
title('Feature Importance: Female (Cluster 2)');
ylabel('Importance Score');
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