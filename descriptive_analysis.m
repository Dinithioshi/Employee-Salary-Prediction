clc;
clear;
file = 'Employee data.csv';
data = readtable(file);

%% 
data.bdate = datetime(data.bdate,'InputFormat','MM/dd/yyyy');
data.age = 1995 - year(data.bdate);
%% 
data.id = [];
data.bdate = [];
%%
data = data(~isnan(data.age),:);
data = data(data.prevexp > 0,:);
%% 
rng(1); 
cv = cvpartition(height(data),'HoldOut',0.2);
trainData = data(training(cv),:);
testData  = data(test(cv),:);

%% Salary Outliers
figure;
boxplot(trainData.salary);
title('Outliers: Salary');

salaryOutliers = sum(isoutlier(trainData.salary));
disp(['Number of outliers in Training Salary: ', num2str(salaryOutliers)]);

%% Univariate Analysis
%% Histogram for Salary
figure;
histogram(trainData.salary);
title('Distribution of Current Salary');
xlabel('Salary ($)');
ylabel('Frequency');

%% Histogram for Age
figure;
histogram(trainData.age);
title('Distribution of Age');
xlabel('Age');
ylabel('Frequency');

%% Bar chart for Job Category
figure;
tab_job = tabulate(trainData.jobcat);
bar(tab_job(:,1), tab_job(:,2));
title('Employee Count by Job Category');
xlabel('1: Clerical, 2: Custodial, 3: Manager');
ylabel('Count');

%% Univariate Graph for Job Time

figure;
histogram(trainData.jobtime);
title('Distribution of Job Time (Months since Hire)');
xlabel('Months since Hire');
ylabel('Frequency');
%% Bar chart for Gender
figure;
categorical_gender = categorical(trainData.gender);
histogram(categorical_gender);
title('Employee Count by Gender');
ylabel('Count');

%% Minority (Bar / Categorical)
figure;
minority_cat = categorical(trainData.minority,[0 1],{'Non-Minority','Minority'});
histogram(minority_cat);
title('Distribution of Minority Status');
ylabel('Count');

%% Previous Experience (Histogram)
figure;
histogram(trainData.prevexp);
title('Previous Experience (Months)');
xlabel('Months');
ylabel('Frequency');

%% Starting Salary (Histogram)
figure;
histogram(trainData.salbegin);
title('Starting Salary Distribution');
xlabel('Beginning Salary ($)');
ylabel('Frequency');

%% Education (Histogram)
figure;
histogram(trainData.educ,'BinMethod','integers');
title('Educational Level (Years)');
xlabel('Years of Education');
ylabel('Frequency');

%% Bivariate Analysis

%% Education vs Salary (Gender Colored)
figure;
gscatter(trainData.educ, trainData.salary, trainData.gender);
title('Salary vs Education');
xlabel('Years of Education');
ylabel('Current Salary ($)');
grid on;

%% Salary by Job Category
figure;
boxplot(trainData.salary, trainData.jobcat);
title('Salary Distribution by Job Category');
xlabel('Job Category');
ylabel('Salary ($)');

%% Salary vs Previous Experience
figure;
scatter(trainData.prevexp, trainData.salary);
title('Salary vs Previous Experience');
xlabel('Prev Experience (Months)');
ylabel('Current Salary ($)');
lsline;

%% Salary vs Beginning Salary
figure;
scatter(trainData.salbegin, trainData.salary);
title('Salary vs Beginning Salary');
xlabel('Beginning Salary ($)');
ylabel('Current Salary ($)');
lsline;

%% Salary by Minority Status
figure;
boxplot(trainData.salary, trainData.minority);
title('Salary by Minority Status');
ylabel('Salary ($)');
xticklabels({'Non-Minority','Minority'});

%% Salary by Gender
figure;
boxplot(trainData.salary, trainData.gender);
title('Salary by Gender');
ylabel('Salary ($)');

%% Salary vs Job Time
figure;
scatter(trainData.jobtime, trainData.salary);
title('Salary vs Months since Hire');
xlabel('Months since Hire');
ylabel('Current Salary ($)');
lsline;

%% Salary vs Age
figure;
scatter(trainData.age, trainData.salary);
title('Salary vs Age');
xlabel('Age (Years)');
ylabel('Current Salary ($)');
lsline;

%% Correlation Heatmap
numericVars = {'age','educ','salary','salbegin','jobtime','prevexp'};
corrMatrix = corr(trainData{:,numericVars},'Rows','complete');
figure;
heatmap(numericVars, numericVars, corrMatrix);
title('Correlation Matrix of Job Factors');
colormap parula;




