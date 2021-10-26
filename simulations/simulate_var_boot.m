%==========================================================================
%  Simulation study of state-space model with switching dynamics 
%==========================================================================


clc; 
clearvars; close all;

%@@@@@@@@ Simulation parameters
M = 2; % number of regimes
p = 2; % VAR order
Ngrid = [10,50,100]; % time series dimension (# variables) 
Tgrid = [400,600,800,1000]; % time series length
nN = numel(Ngrid);
nT = numel(Tgrid);
NTgrid = [repelem(Ngrid,1,nT); repmat(Tgrid,1,nN)];
nrep = 500; % number of replications for each (N,T)
B = 500; % number of bootstrap replicates
nlags = 5; % number of lags at which to evaluate estimation
parallel = true;
conf_level = 0.9;
npars_ci = 6; % number of parameters to build CI for: A, Q, Z, ACF, COV, COR
nboot_ci = 3; % number of bootstrap CI types: percentile, basic, normal
coverage = NaN(nboot_ci,npars_ci,nrep);
% rng(14) % set seed for reproducibility

% N = 50;
% T = 800;



%-------------------------------------------------------------------------%
%            PRIMARY SIMULATION LOOP OVER DATA DIMENSIONS N,T             %
%-------------------------------------------------------------------------%

warning("off")

for i = 1:nN*nT
    tic

    N = NTgrid(1,i);
    T = NTgrid(2,i); 
    fprintf('Simulations for N=%d T=%d\n',N,T);
    outfile = sprintf('result_sim_var_boot_N%dT%d.mat',N,T);



%-------------------------------------------------------------------------%
%                   Generate model parameters                             %
%-------------------------------------------------------------------------%

    % Must ensure that: (i) A is stable, (ii) SNR not too low at
    % observation level

stable = false;
snr = false;
while ~stable || ~snr
    % Transition matrices for dynamics
    A = zeros(N,N,p,M);
%     s = round(.02*N^2);
    for j = 1:M
        sp = zeros(N);
%         sp(randi(N^2,s,1)) = 0.1 * randn(s,1);
        A(:,:,1,j) = diag(rand(N,1) * .1 + .85) + sp;
%         sp = zeros(N);
%         sp(randi(N^2,s,1)) = 0.1 * randn(s,1);
        A(:,:,2,j) = diag(rand(N,1) * .1 - .05) + sp;
    end


    % Check stationarity
    if p == 1
        Abig = zeros(N);
    else
        Abig = diag(ones((p-1)*N,1),-N);
    end
    test = false(M,1);
    for j = 1:M
        Abig(1:N,:) = reshape(A(:,:,:,j),[N,p*N]); 
        eigA = eig(Abig);
        test(j) = all(abs(eigA) <= .99);
    end
    stable = all(test);
    if ~stable 
        continue
    end

    % State noise covariance matrix
    sig2Q = 1e-3/N;
    %         Q = repmat(sig2Q * eye(r),1,1,M);
    Q = zeros(N,N,M);
    for j = 1:M
        Q_j = randn(N);
        Q(:,:,j) = sig2Q * (Q_j' * Q_j);
    end

    % Stationary variance matrices and autocorrelation functions
    theta = struct('A',A,'Q',Q);
    [ACF,~,COV] = get_covariance(theta,nlags,0);

    % Test SNR
    signal = zeros(1,M); 
    noise = zeros(1,M);
    for j = 1:M
        signal(j) = sum(diag(COV(:,:,j)));
        noise(j) = sum(diag(Q(:,:,j)));
    end
    signal = signal - noise;
    snr = all(signal >= 5 * noise & signal <= 10 * noise);        
end


% Initial means and variances
theta.mu = zeros(N,M);
theta.Sigma = repmat(0.1 * eye(N),1,1,M);
% Initial regime probabilities
theta.Pi = [1;0];
% Transition probability matrix
theta.Z = [.98,.02;.02,.98];

% Misc
ACF = ACF(:,2:nlags+1,:);
ACF = ACF(:);
mask_COV = find(tril(ones(N)));
mask_COR = find(tril(ones(N),-1));
COR = zeros(N,N,M);
for j = 1:M
    COR(:,:,j) = corrcov(COV(:,:,j) + COV(:,:,j)');
end
alpha = (1-conf_level); 
q_alpha = norminv(1-alpha/2);
COV = reshape(COV,N^2,M);
COV = COV(mask_COV,:);
COV = COV(:);
COR = reshape(COR,N^2,M);
COR = COR(mask_COR,:);
COR = COR(:);

% Control parameters for EM
control = struct('eps',1e-6,'ItrNo',200,'beta0',1,'betarate',1,...
    'verbose',false);
control2 = control; control2.ItrNo = 800;
control_boot = struct('eps',1e-6,'ItrNo',100,'beta0',1,...
    'betarate',1.01,'verbose',false);

% Fixed coefficients
fixed = struct('A',repmat(diag(NaN(N,1)),[1,1,p,M]));  

% Equality constraints 
equal = struct('mu',true,'Sigma',true);

% Scale of (columns of) matrices C(j) and/or upper bound for 
% eigenvalues of A(j)
scale = struct('A',.999);  

% EM initialization
opts = struct('segmentation','fixed','len',50);

clear Abig Q_j eigA signal noise snr Aboot ACFboot Qboot Zboot Sigma
   


%-------------------------------------------------------------------------%
%                   SECONDARY LOOP OVER REPLICATIONS                      %
%-------------------------------------------------------------------------%


A = A(:);
Q = reshape(Q,N^2,M);
Q = Q(mask_COV,:);
Q = Q(:);
Z = theta.Z(:);
target = {A,Q,Z,ACF,COV,COR};

Ahat = NaN(N*N*p*M,nrep);
Qhat = NaN(N*(N+1)/2*M,nrep);
COVhat = NaN(N*(N+1)/2*M,nrep);
CORhat = NaN(N*(N-1)/2*M,nrep);
mean_boot_A = NaN(N*N*p*M,nrep);
mean_boot_Q = NaN(N*(N+1)/2*M,nrep);
mean_boot_COV = NaN(N*(N+1)/2*M,nrep);
mean_boot_COR = NaN(N*(N-1)/2*M,nrep);
sd_boot_A = NaN(N*N*p*M,nrep);
sd_boot_Q = NaN(N*(N+1)/2*M,nrep);
sd_boot_COV = NaN(N*(N+1)/2*M,nrep);
sd_boot_COR = NaN(N*(N-1)/2*M,nrep);

% Permutations for matching
sigma = perms(1:M); 
factM = factorial(M);

for rep = 1:nrep
    fprintf("Replication %d\n",rep)



%-------------------------------------------------------------------------%
%                           Generate data                                 %
%-------------------------------------------------------------------------%


    [y,S] = simulate_var(theta,T);



%-------------------------------------------------------------------------%
%                         Fit switching SSM                               %
%-------------------------------------------------------------------------%


    % Initialization
    pars0 = init_var(y,M,p,opts,control,equal,fixed,scale);

    % EM algorithm           
     try
        [~,~,~,Shat,pars,LL] = ... 
            switch_var(y,M,p,pars0,control,equal,fixed,scale);
     catch
        continue                    
     end

    % Acceleration
    try 
         pars1 = fast_var(y,M,p,Shat,control2,equal,fixed,scale);
         [~,~,~,~,pars2,LL2] = ... 
            switch_var(y,M,p,pars1,control,equal,fixed,scale);
         if max(LL2) > max(LL)
             pars = pars2;
         end
    catch
    end
    
    % Stationary, regime-specific covariance and autocorrelation 
    [ACFhat_tmp,~,COVhat_tmp] = get_covariance(pars,nlags,0);
    ACFhat_tmp = reshape(ACFhat_tmp(:,2:nlags+1,:),N*nlags,M);
    CORhat_tmp = NaN(N,N,M);
    for j = 1:M
        CORhat_tmp(:,:,j) = ...
            corrcov(COVhat_tmp(:,:,j) + COVhat_tmp(:,:,j)');
    end
    COVhat_tmp = reshape(COVhat_tmp,N^2,M);
    COVhat_tmp = COVhat_tmp(mask_COV,:);
    CORhat_tmp = reshape(CORhat_tmp,N^2,M);
    CORhat_tmp = CORhat_tmp(mask_COR,:);
   
    % Match estimates to model parameters by regime
    classif = zeros(1,factM);    
    for m = 1:factM
        S_perm = sigma(m,Shat);                           
        classif(m) = mean(S_perm == S);
%         d(m) = mean(abs(COV-COVhat_tmp(:,sigma(m,:))),'all');
    end
    [~,idx] = min(classif);
    sigma_best = sigma(idx,:);
    if ~isequal(sigma_best,1:M)
        COVhat_tmp = COVhat_tmp(:,sigma_best);
        CORhat_tmp = CORhat_tmp(:,sigma_best);
        ACFhat_tmp = ACFhat_tmp(:,sigma_best);
        pars.A = pars.A(:,:,:,sigma_best);
        pars.Q = pars.Q(:,:,sigma_best);
        pars.mu = pars.mu(:,sigma_best);
        pars.Sigma = pars.Sigma(:,:,sigma_best);
        pars.Pi = pars.Pi(sigma_best);
        pars.Z = pars.Z(sigma_best,sigma_best);
    end
     
    Ahat_tmp = pars.A(:);
    Ahat(:,rep) = Ahat_tmp; 
    Qhat_tmp = reshape(pars.Q,[],M);
    Qhat_tmp = reshape(Qhat_tmp(mask_COV,:),[],1);
    Qhat(:,rep) = Qhat_tmp; 
    COVhat(:,rep) = COVhat_tmp(:);
    CORhat_tmp = CORhat_tmp(:);
    CORhat(:,rep) = CORhat_tmp;
    ACFhat_tmp = ACFhat_tmp(:);
 



%-------------------------------------------------------------------------%
%                               Bootstrap                                 %
%-------------------------------------------------------------------------%


    % Parametric bootstrap
    [parsboot,LLboot] = bootstrap_var(pars,T,B,opts,...
            control_boot,equal,fixed,scale,parallel);
    % Nonparametric boostrap
%     [parsboot,LLboot] = bootstrap_var_npar(pars,y,S,B,opts,...
%         control_boot,equal,fixed,scale,parallel);

        
    % Target of boostrap inference: A, Q, Z, COV, COR, ACF
    % Bootstrap CI methods: 1) percentile, 2) basic, 3) normal  
    Aboot = parsboot.A;
    Qboot = parsboot.Q;
    Zboot = parsboot.Z;
    ACFboot = NaN(N,nlags,M,B);
    COVboot = NaN(N*(N+1)/2,M,B);
    CORboot = NaN(N*(N-1)/2,M,B);
    for b = 1:B
        parsb = struct('A',Aboot(:,:,:,:,b), ...
            'Q',Qboot(:,:,:,b));                 
        [ACFb,~,COVb] = get_covariance(parsb,nlags,0);
        CORb = NaN(N,N,M);
        for j = 1:M
            CORb(:,:,j) = corrcov(COVb(:,:,j)+COVb(:,:,j)');
        end
        ACFboot(:,:,:,b) = ACFb(:,2:end,:);
        COVb = reshape(COVb,N^2,M);        
        COVboot(:,:,b) = COVb(mask_COV,:);
        CORb = reshape(CORb,N^2,M);        
        CORboot(:,:,b) = CORb(mask_COR,:);
    end

    % Reshape bootstrap estimates
    Aboot = reshape(Aboot,N^2*p,M,B);
    Qboot = reshape(Qboot,N^2,M,B);
    Qboot = Qboot(mask_COV,:,:);
    ACFboot = reshape(ACFboot,N*nlags,M,B);

    % Match bootstrap regimes to regimes of initial estimate 
    % (permute indices as needed)
    d = zeros(factM,B);
    for m = 1:factM
        e = COVboot(:,sigma(m,:),:) - COVhat_tmp;
        d(m,:) = squeeze(sum(abs(e),1:2));
    end
    [~,idx] = min(d); % permutation yielding the best match
    COVhat_tmp = COVhat_tmp(:);
    for m = 1:factM
        if isequal(sigma(m,:),1:M)
            continue
        end
        idx2 = find(idx == m); 
        % If best permutation is not identity matrix, apply it to
        % bootstrap parameter estimates
        if ~isempty(idx2) 
            Aboot(:,:,idx2) = Aboot(:,sigma(m,:),idx2);
            Qboot(:,:,idx2) = Qboot(:,sigma(m,:),idx2);
            ACFboot(:,:,idx2) = ACFboot(:,sigma(m,:),idx2);
            COVboot(:,:,idx2) = COVboot(:,sigma(m,:),idx2);
            CORboot(:,:,idx2) = CORboot(:,sigma(m,:),idx2);
            Zboot(:,:,idx2) = Zboot(sigma(m,:),sigma(m,:),idx2);                
        end
    end


%-------------------------------------------------------------------------%
%                     Bootstrap: Pointwise inference                      %
%-------------------------------------------------------------------------%


    Aboot = reshape(Aboot,[],B);
    Qboot = reshape(Qboot,[],B);
    Zboot = reshape(Zboot,[],B);
    ACFboot = reshape(ACFboot,[],B);
    COVboot = reshape(COVboot,[],B);
    CORboot = reshape(CORboot,[],B);
    
    mle = {Ahat_tmp,Qhat_tmp,pars.Z(:),ACFhat_tmp,COVhat_tmp,CORhat_tmp};
    boot = {Aboot,Qboot,Zboot,ACFboot,COVboot,CORboot};
    
      coverage_tmp = NaN(nboot_ci,npars_ci);
        
        for k = 1:npars_ci
            % Summary statistics
            mean_boot = mean(boot{k},2,'omitNaN');
            sd_boot = std(boot{k},1,2,'omitNaN');
            qt1_boot = quantile(boot{k},alpha/2,2);
            qt2_boot = quantile(boot{k},1-alpha/2,2);
            switch k
                case 1
                    mean_boot_A(:,rep) = mean_boot;
                    sd_boot_A(:,rep) = sd_boot;
                case 2
                    mean_boot_Q(:,rep) = mean_boot;
                    sd_boot_Q(:,rep) = sd_boot;
                case 5
                    mean_boot_COV(:,rep) = mean_boot;
                    sd_boot_COV(:,rep) = sd_boot;
                case 6
                    mean_boot_COR(:,rep) = mean_boot;
                    sd_boot_COR(:,rep) = sd_boot;
            end
            
            % Percentile bootstrap CI
            coverage_tmp(1,k) = ...
                mean(qt1_boot <= target{k} & target{k} <= qt2_boot);
            % Basic bootstrap CI
            coverage_tmp(2,k) = mean(2 * mle{k} - qt2_boot <= target{k} & ...
                target{k} <= 2 * mle{k} - qt1_boot);
            % Normal bootstrap CI
            coverage_tmp(3,k) = mean(abs(2 * mle{k} - mean_boot - target{k})...
                <= q_alpha * sd_boot);
        end
        
        coverage(:,:,rep) = coverage_tmp;
        fprintf("Current coverage level\n");
        tmp = array2table(mean(coverage(:,:,1:rep),3,'omitNaN'),...
            'RowNames',["pct","basic","norm"],...
            "VariableNames",["A","Q","Z","ACF","COV","COR"]);
        disp(tmp);

        % Clean up and save results periodically 
        clear mle pars0 pars1 pars2 pars Ahat_tmp Qhat_tmp
        clear ACFhat_tmp COVhat_tmp CORhat_tmp 
        clear parsboot boot Aboot Qboot Zboot ACFboot COVboot CORboot
        clear mean_boot sd_boot qt1_boot qt2_boot COVb CORb ACFb parsb
        clear d e y S LL LL2 sigma_best
        
        if mod(rep,10) == 0
            save(outfile)
        end
    
end

    
toc
    
% clear COVb d e idx idx2 factM j sigma 
% clear ACFhat_tmp 
% clear LL LL2 pars0 pars1 pars2 Ahat_tmp ACFb b parsboot Qhat_tmp test lb ub y
% clear COVhat_tmp pars parsb Qtmp sigma_best 




%-------------------------------------------------------------------------%
%                             Save results                                %
%-------------------------------------------------------------------------%

save(outfile)

    
end




% i1 = 8;
% i2 = 5;
% j = 1;
% rep = 3;
% tmp = squeeze(COVhat(i1,i2,j,:));
% tiledlayout(2,2);
% nexttile
% qqplot(tmp);
% title("QQ plot estimator")
% nexttile
% histogram(tmp);
% title("Histogram estimator")
% tmp = squeeze(COVboot(i1,i2,j,:,rep));
% nexttile
% qqplot(tmp);
% title("QQ plot Bootstrap")
% nexttile
% histogram(tmp);
% title("Histogram Bootstrap")
% 
% 
% %%
% 
% meanhat = mean(COVhat,4);
% sdhat = std(COVhat,0,4);
% biashat = meanhat - COV;
% 
% mean(abs(COVhat - biashat - COV) <= 1.96 * sdhat,4)
% 
% lb = squeeze(quantile(COVboot,(1-conf_level)/2,4));
% ub = squeeze(quantile(COVboot,(1+conf_level)/2,4));
% 
% mean(lb <= COV & COV <= ub,4)
% 
% 
% mean(2 * COVhat - ub <= COV & COV <= 2 * COVhat - lb,'all')
% 
% meanboot = squeeze(mean(COVboot,4));
% biasboot =  meanboot - COVhat;
% sdboot = squeeze(std(COVboot,0,4));
% mean(abs(COVhat - biasboot - COV) <= 1.96 * sdhat,4)

% tmp = zeros(N*N,M);
% tmp(mask,:) = COV;
% tmp = reshape(tmp,N,N,M);
% for j = 1:M
%     tmp(:,:,j) = tmp(:,:,j) + (tril(tmp(:,:,j),-1)');
% end

%%

estim = Qhat;
targt = Q(:); 
meanhat = mean(estim,2);
sdhat = std(estim,0,2);
biashat = meanhat - targt;
mean(abs(estim - biashat - targt) <= 1.96 * sdhat,'all')


%%

close all
tiledlayout(5,10);
idx = randperm(size(Ahat,1),50);
for i = 1:50
    nexttile
    qqplot(Ahat(i,:));
    title([])
    xlabel([])
    ylabel([])
end
