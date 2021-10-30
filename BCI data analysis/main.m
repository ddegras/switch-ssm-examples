%=========================================================================%
%          FIT SWITCHING DYNAMICS MODEL AND CALCULATE DFC MEASURES        %
%=========================================================================%


% Make sure path to data folder is set in MATLAB
clc; clearvars;

% Subject IDs
subs = ["a","b","f","g"];
nsubs = length(subs);

% Session IDs
sess = ["calib" "eval"];
nsess = length(sess);

% Model type and hyperparameters
model = 'dyn'; 
M = 5; 
p = 3; 
r = 5; 
N = 59;

% EM control parameters
verbose = true;
control = struct('eps',1e-4,'ItrNo',50,'beta0',0.5,'betarate',1.02,... 
    'abstol',1e-8,'reltol',1e-8,'safe',false,'verbose',true); 
control2 = struct('eps',5e-6,'ItrNo',500,... 
    'abstol',1e-8,'reltol',1e-8,'safe',false,'verbose',true);
opts = struct('segmentation','fixed','len',1000,'Replicates',20);
ncycles = 1; % 5;

% Result objects
unsupervised_fit = cell(nsubs,nsess); % estimated model parameters
supervised_fit = cell(nsubs,nsess); % estimated model parameters
dfc_measures = cell(nsubs,nsess); % dynamic FC components


for i = 1:nsubs

    subject = subs{i};
    
    for k = 1:nsess
       
        session = sess{k};        
        fprintf('\n\nSubject %s Session %s',upper(subject),...
            upper(session));
        
        % Load data
        fname = sprintf('BCICIV_%s_ds1%s.mat',session,subject);
        load(fname)
        
        
        %=================
        % Preprocess data
        %=================
        
        % For each EEG time series, apply bandpass filter  
        % with pass range [8Hz,25Hz], downsample to 50Hz, clip extreme 
        % observations (more than 10 standard deviations way from mean), 
        % and standardize (mean zero, variance 1)
        % Also create time vector of task labels 
        % (1 = task 1, 2 = rest, 3 = task 2)
        switch session 
            case 'calib'
                [y,S] = preprocess_bci(cnt,mrk);
            case 'eval'
                y = preprocess_bci(cnt);
                fname2 = sprintf('BCICIV_eval_ds1%s_1000Hz_true_y',subject);
                S = load(fname2);
                S = downsample(S.true_y,20);
                S = S(1:size(y,2));
                S = S + 2;
        end
                
        % Relabel any missing value in stimulus sequence as extra task (#4) 
        test = isnan(S);
        ntasks = length(unique(S(~test)));
        if any(test)
            S(test) = ntasks + 1;
            ntasks = ntasks + 1;
        end
        
        
        %================================================
        % Fit switching dynamics model (unsupervised fit)
        %================================================

        out1 = fit_bci(y,model,M,p,r,opts,control,control2,ncycles);
        out1.subject = subject; out1.session = session; out1.task = S;
        unsupervised_fit{i,k} = out1; 
        
        
        %==============================================
        % Fit switching dynamics model (supervised fit)
        %==============================================        
        
        [~,~,pars,LL] = fast_dyn(y,ntasks,p,r,S,[],control2);        
        out2 = struct();
        out2.subject = subject; out2.session = session; 
        out2.pars = pars; out2.LL = max(LL); out2.task = S;
        supervised_fit{i,k} = out2;        
                   
        
        %========================
        % Calculate DFC measures
        %========================

        lagmax = 50;
        out = out1; 
        stationary = get_covariance(out.pars,lagmax,0);
        dwell = tabulate(out.regime);
        out = struct('subject',subject,'session',session,'M',M,'p',p,...
            'r',r,'ACF',stationary.ACF,'COV',stationary.COV,...
            'COR',stationary.COR,'PCOR',stationary.PCOR,'dwell',dwell);
        dfc_measures{i,k} = out;
        
    end
end

clear cnt dwell out* S stationary y

% Save results
save('unsupervised_pars.mat','unsupervised_fit')
save('supervised_pars.mat','supervised_fit')
save('dfc_measures.mat','dfc_measures')



%%



%=========================================================================%
%                         CLUSTER DFC MEASURES                            %
%=========================================================================%


% FIGURE 3 OF SUPPLEMENTARY MATERIALS


% Prepare DFC measure for clustering 
dfc = zeros(N*(N+1)/2,M,nsubs,nsess);
mask = repmat(logical(tril(ones(N))),1,1,M);
dwell = zeros(M,nsubs,nsess);
for session = 1:nsess
    for sub = 1:nsubs
        dfc(:,:,sub,session) = ...
            reshape(dfc_measures{sub,session}.COV(mask),[],M);
        dwell(:,sub,session) = dfc_measures{sub,session}.dwell(:,3) / 100;
    end
end
dfc = reshape(dfc,N*(N+1)/2,M*nsubs*nsess);
dfc = dfc ./ sqrt(sum(dfc.^2));

% Calculate clustering distances
metric = 'squaredeuclidean';
z = linkage(dfc','complete',metric);
dendrogram(z)

% Hierarchical clustering
theta = 8;
thres = 2 * (1 - cosd(theta));
clust = cluster(z,'Cutoff',thres,'Criterion','distance');

% Cluster number and sizes
tab = tabulate(clust);
nclusters = size(tab,1);
cluster_size = tab(:,2);

% Visualize clusters
close all
load('BCICIV_calib_ds1a.mat', 'nfo')
channel = nfo.clab;
valid = find(cluster_size >= 3); % sufficiently large clusters
nvalid = length(valid);
for c = 1:nvalid
    figure(c)
    clf   
    idx = find(clust == tab(valid(c),1));
    [regime,subject,session] = ind2sub([M,nsubs,nsess],idx);
    nr = ceil(cluster_size(valid(c))/5);
    if nr == 1
        nc = cluster_size(valid(c));
    else
        nc = 5;
    end
    tiledlayout(nr,nc,'Padding','compact')    
    for i = 1:cluster_size(valid(c))
        nexttile
        imagesc(dfc_measures{subject(i),session(i)}.COV(:,:,regime(i)));
        xticklabels([]);
%         xticks(1:N); xticklabels(channel); xtickangle(90); 
        yticks(1:N); yticklabels(channel);
        title_i = sprintf('%s-%d-%s (%2.1f%%)',subs{subject(i)}, ...
            regime(i), sess{session(i)}, ...
            dwell(regime(i),subject(i),session(i))*100);
        title(title_i,'FontSize',12)
        colorbar
    end
end


% % SAVE COVARIANCE PLOTS AS PDF
% mkdir FIGURES
% wordkdir = pwd;
% f = figure;
% f.PaperUnits = 'centimeters';
% f.PaperSize = [12 10];
% 
% % Autocorrelation plots
% f.PaperPosition = [0,0.05,12,9.95];
% for j = 1:M
%     plot(out.ACF(:,:,j)')
%     xlabel('Lag')
%     ylabel('Autocorrelation')
%     title(sprintf('Regime %d (%.1f%%)',j,dwell(j,3)))
%     figname = sprintf('%s/FIGURES/ACF_%s_%s_%d.pdf',workdir,...
%         session,subject,j);
%     saveas(f,figname);
% end       
% 
% % Covariance, correlation, and partial correlation plots 
% fc_type = {'COV','COR','PCOR'};
% f.PaperPosition = [0,0,12,10];
% channel = nfo.clab;
% for l = 1:3
%     for j = 1:M
%         imagesc(out.(fc_type{l})(:,:,j))
%         colorbar
%         ax = gca;
%         ax.XAxis.FontSize = 4.5;
%         ax.YAxis.FontSize = 4.5;
%         xticks(1:N); yticks(1:N);
%         xticklabels(channel); xtickangle(90); yticklabels(channel); 
%         title(sprintf('Regime %d (%.1f%%)',j,dwell(j,3)))
%         figname = sprintf('%s/FIGURES/%s_%s_%s_%d.pdf',workdir,...
%             fc_type{l},session,subject,j);
%         saveas(f,figname);
%     end
% end


%% 


%=========================================================================%
%          STUDY VARIATIONS IN DFC BETWEEN SUBJECTS AND SESSIONS          %
%=========================================================================%


% TABLE 1 OF SUPPLEMENTARY MATERIALS

dfc = reshape(dfc,N*(N+1)/2,M,nsubs,nsess);
table1 = zeros(nsess * nsubs); 
for k = 1:nsess
    for i = 1:nsubs
        for j = i:nsubs
            if i == j
                table1((k-1)*nsubs+i,(k-1)*nsubs+j) = ...
                    distfun(dfc(:,:,i,k),dwell(:,i,k));
            else
                table1((k-1)*nsubs+i,(k-1)*nsubs+j) = ...
                    distfun(dfc(:,:,i,k),dwell(:,i,k),...
                    dfc(:,:,j,k),dwell(:,j,k));
            end
        end
    end
end

for i = 1:nsubs
    for j = 1:nsubs
            table1(i,nsubs+j) = distfun(dfc(:,:,i,1),dwell(:,i,1),...
                dfc(:,:,j,2),dwell(:,j,2));
    end
end

labels = compose("%s-%s",repelem(upper(sess(:)),nsubs),...
    repmat(upper(subs(:)),nsess,1));


table1 = cell2table(num2cell(table1),'RowNames',labels,'VariableNames',labels);
disp(table1);



%% 


%=========================================================================%
%                   FC REGIMES AND EXPERIMENTAL TASKS                     %
%=========================================================================%



% TABLE 2 OF SUPPLEMENTARY MATERIALS

session = 1; % calibration 
ntasks = 3; 
tasks = struct('a',["left";"foot";"rest"], 'b', ["left";"right";"rest"], ...
    'f',["left";"foot";"rest"], 'g', ["left";"right";"rest"]);

table2 = zeros(nsubs*ntasks,M);
for i = 1:nsubs
    idx = (i-1)*ntasks+1:i*ntasks;
    table2(idx,:) = crosstab(unsupervised_fit{i,session}.task,...
        unsupervised_fit{i,session}.regime);
    table2(idx,:) = table2(idx,:) ./ sum(table2(idx,:)) * 100; 
end
labels1 = repelem(upper(subs(:)),ntasks); 
labels2 = string(1:nsubs*ntasks);
for i = 1:nsubs
    idx = (i-1)*ntasks+1:i*ntasks;
    labels2(idx) = tasks.(subs(i));
end
labels2 = labels2(:);
rlabels = compose("%s-%s",labels1,labels2);
labels1 = repelem("Regime",M);
labels2 = 1:M;
clabels = compose("%s %d",labels1(:),labels2(:));
table2 = cell2table(num2cell(table2),'RowNames',rlabels,'VariableNames',clabels);
disp(table2);




