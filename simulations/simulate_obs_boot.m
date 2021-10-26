%==========================================================================
%  Simulation study of state-space model with switching dynamics 
%==========================================================================


clc; 
clearvars; close all;

%@@@@@@@@ Simulation parameters
M = 2; % number of regimes
p = 2; % VAR order
r = 2;  % state vector dimension
Ngrid = [10,50,100]; % time series dimension (# variables) 
Tgrid = [400,600,800,1000]; % time series length
nN = numel(Ngrid);
nT = numel(Tgrid);
NTgrid = [repelem(Ngrid,1,nT); repmat(Tgrid,1,nN)];
nrep = 200; % number of replications for each (N,T)
B = 100; % number of bootstrap replicates
nlags = 5; % number of lags at which to evaluate estimation
parallel = true;
conf_level = 0.9;
npars_ci = 6; % number of parameters to build CI for: COV, COR, ACF, PC, Z, R
nboot_ci = 3; % number of bootstrap CI types: percentile, basic, normal
% rng(14) % set seed for reproducibility





%-------------------------------------------------------------------------%
%            PRIMARY SIMULATION LOOP OVER DATA DIMENSIONS N,T             %
%-------------------------------------------------------------------------%

for i = 6:nN*nT

    tic
    N = NTgrid(1,i);
    T = NTgrid(2,i); 


    % fprintf('Simulations for N=%d T=%d\n',N,T);
    warning("off")

    % Output structures
    coverage = NaN(nboot_ci,npars_ci,nrep);
    Rhat = NaN(N*(N+1)/2,nrep);
    Zhat = NaN(M^2,nrep);
    ACFhat = NaN(N*nlags*M,nrep);
    COVhat = NaN(N*(N+1)/2*M,nrep);
    CORhat = NaN(N*(N-1)/2*M,nrep);

    % Output structures for bootstrap 
    % Useful to assess issues in bootstrap CI coverage
    % However these summaries alone cannot pick up on problems such as multimodality
    % Index 1: component in parameter
    % Index 2: summary statistic: 1 = mean, 2 = sd, 3 = lower quantile (alpha/2), 
    %           4 = upper quantile (1 - alpha/2)
    % Index 3: replication
    summary_boot_PC = NaN(N*(N+1)/2*M,4,nrep);
    summary_boot_R = NaN(N*(N+1)/2,4,nrep);
    summary_boot_Z = NaN(M^2,4,nrep);
    summary_boot_ACF = NaN(N*nlags*M,4,nrep);
    summary_boot_COV = NaN(N*(N+1)/2*M,4,nrep);
    summary_boot_COR = NaN(N*(N-1)/2*M,4,nrep);

    outfile = sprintf('result_sim_obs_boot_N%dT%d.mat',N,T);



%-------------------------------------------------------------------------%
%                   Generate model parameters                             %
%-------------------------------------------------------------------------%

    % Must ensure that: (i) A is stable, (ii) SNR not too low at
    % observation level

    stable = false;
    snr = false;
    while ~stable || ~snr
        % Transition matrices for state vector dynamics
        A = rand(r,r,p,M); 
        A(:,:,1,:) = A(:,:,1,:) * .7; 
        A(:,:,2,:) = A(:,:,2,:) * .3; 
        % Check stationarity
        Abig = diag(ones((p-1)*r,1),-r);
        test = false(M,1);
        for j = 1:M
            Abig(1:r,:) = reshape(A(:,:,:,j),[r,p*r]);
            eigA = eig(Abig);
            test(j) = all(abs(eigA) <= .99);
        end
        stable = all(test);
        if ~stable 
            continue
        end

        % Observation matrix    
        C = NaN(N,r,M);
        for j = 1:M
            [Cj,~,~] = svd(randn(N,r),'econ');
            C(:,:,j) = Cj;
        end

        % State noise covariance matrix
        sig2Q = 5e-3;
        % Q = repmat(sig2Q * eye(r),1,1,M);
        Q = zeros(r,r,M);
        for j = 1:M
            Q_j = randn(r);
            Q(:,:,j) = sig2Q * (Q_j' * Q_j);
        end

        % Observation noise covariance matrix
        sig2R = 5e-3 / N;
        rhoR = 0.1; % exchangeable noise structure
        R = rhoR * sig2R * ones(N) + sig2R * (1-rhoR) * eye(N);

        % Gather model parameters
        theta = struct('A',A, 'C',C, 'Q',Q, 'R',R);

        % Stationary variance matrices and autocorrelation functions
        [ACF,~,COV] = get_covariance(theta,nlags,0);

        % Test SNR
        signal = zeros(1,M); 
        for j = 1:M
            signal(j) = sum(diag(COV(:,:,j))) - sum(diag(R));
        end
        noise = sum(diag(R));
        snr = all(signal >= 5 * noise & signal <= 10 * noise);        

    end

    clear signal noise snr test stable

    % Initial means and variances
    theta.mu = zeros(r,M);
    theta.Sigma = repmat(0.1 * eye(r),1,1,M);
    % Initial regime probabilities
    theta.Pi = [1;0];
    % Transition probability matrix
    Z = [.98,.02;.02,.98];
    theta.Z = Z;

    % Masks 
    mask_ACF = true(N,nlags+1,M);
    mask_ACF(:,1,:) = false;
    mask_ACF = mask_ACF(:);
    mask_R = find(logical(tril(ones(N))));
    mask_COV = logical(repmat(tril(ones(N)),[1,1,M]));
    mask_COR = logical(repmat(tril(ones(N),-1),[1,1,M]));

    % Reshape model parameters
    A = A(:);
    PC = zeros(N,N,M);
    for j = 1:M
        Cj = C(:,:,j);
        PC(:,:,j) = Cj*((Cj'*Cj)\(Cj'));
    end
    PC = PC(mask_COV);
    R = R(mask_R);
    Z = Z(:);

    % Misc
    alpha = 1 - conf_level;
    q_alpha = norminv(1-alpha/2); % quantile for normal CI

    % Covariance-related quantities
    ACF = ACF(mask_ACF);
    COR = NaN(N,N,M);
    for j = 1:M
        COR(:,:,j) = corrcov(COV(:,:,j)+COV(:,:,j)');
    end
    COV = COV(mask_COV);
    COR = COR(mask_COR);

    % Inference targets
    target = {COV,COR,ACF,PC,Z,R};

    % EM control parameters
    control = struct('eps',1e-6,'ItrNo',50,'beta0',1,'betarate',1,...
        'safe',false,'verbose',false);
    control_boot = struct('eps',1e-6,'ItrNo',50,'verbose',false);
    control2 = control; control2.ItrNo = 500;

    % Fixed coefficients
    fixed = [];  

    % Equality constraints 
    equal = struct('mu',true,'Sigma',true);

    % Scale of (columns of) matrices C(j) and/or upper bound for 
    % eigenvalues of A(j)
    scale = struct('A',.99);  

    % Initialization
    opts = struct('segmentation','fixed','len',50);

    % Turn off warnings
    warning("off")

    % Permutations for matching
    sigma = perms(1:M); 
    factM = factorial(M);




%-------------------------------------------------------------------------%
%                         LOOP OVER REPLICATIONS                          %
%-------------------------------------------------------------------------%

  

    for rep = 1:nrep

        fprintf("Replication %d\n",rep)




%-------------------------------------------------------------------------%
%                           Generate data                                 %
%-------------------------------------------------------------------------%


        [y,S] = simulate_obs(theta,T);



%-------------------------------------------------------------------------%
%                         Fit switching SSM                               %
%-------------------------------------------------------------------------%

        % Initialization
        pars0 = init_obs(y,M,p,r,opts,control,equal,fixed,scale);

         % EM algorithm           
         try
            [~,~,~,Shat,~,~,pars,LL] = ... 
                switch_obs(y,M,p,r,pars0,control,equal,fixed,scale);
         catch
            continue                    
         end
         % Acceleration
         try 
             [~,~,pars1] = ...
                 fast_obs(y,M,p,r,Shat,pars,control2,equal,fixed,scale);
             [~,~,~,Shat2,~,~,pars2,LL2] = ... 
                switch_obs(y,M,p,r,pars1,control,equal,fixed,scale);
             if max(LL2) > max(LL)
                 pars = pars2;
                 Shat = Shat2;
             end
         catch
         end

        % Match estimated regimes to true regimes based on classification
        % rate (formerly: stationary covariance)
        if M > 1
            classif = NaN(factM,1);
            for m = 1:factM
                S_perm = sigma(m,Shat);                           
                classif(m) = mean(S_perm == S);
            end
            [~,idx] = max(classif);
            sigma_best =  sigma(idx,:);

            % Re-arrange estimated regimes and parameters as needed
            if ~isequal(sigma_best,1:M)
                pars.A(:,:,:,sigma_best) = pars.A;
                pars.C(:,:,sigma_best) = pars.C;
                pars.Q(:,:,sigma_best) = pars.Q;
                pars.mu(:,sigma_best) = pars.mu;
                pars.Sigma(:,:,sigma_best) = pars.Sigma;
                pars.Pi(sigma_best) = pars.Pi;
                pars.Z(sigma_best,sigma_best) = pars.Z;
            end
        end

        % Projection matrix on state space
        PChat = NaN(N,N,M); 
        for j = 1:M
            Chatj = pars.C(:,:,j);
            PChat(:,:,j) = Chatj * ((Chatj'*Chatj)\(Chatj')); 
        end
        PChat = PChat(mask_COV);

        % Stationary, regime-specific covariance and autocorrelation 
        [ACFhat_tmp,~,COVhat_tmp,VARhat] = get_covariance(pars,nlags,0);
        ACFhat_tmp = ACFhat_tmp(mask_ACF);
        CORhat_tmp = NaN(N,N,M);
        for j = 1:M
            SDj = sqrt(VARhat(:,j));
            SDj(SDj < eps(1)) = 1;
            CORhat_tmp(:,:,j) = (1 ./ SDj) .* COVhat_tmp(:,:,j) ./ (SDj');
        end
        COVhat_tmp = reshape(COVhat_tmp(mask_COV),[],M); 
        CORhat_tmp = CORhat_tmp(mask_COR);

        % Reshape MLE
        Rhat_tmp = pars.R(mask_R);
        Rhat(:,rep) = Rhat_tmp;
        Zhat_tmp = pars.Z(:);
        Zhat(:,rep) = Zhat_tmp;
        COVhat(:,rep) = COVhat_tmp(:);
        CORhat(:,rep) = CORhat_tmp;
        ACFhat(:,rep) = ACFhat_tmp;





%-------------------------------------------------------------------------%
%                               Bootstrap                                 %
%-------------------------------------------------------------------------%


         [parsboot,LLboot] = bootstrap_obs(pars,T,B,opts,...
                control_boot,equal,fixed,scale,parallel);

        % Target of boostrap inference: COV, COR, ACF, C(C'C)^(-1)C', Z, R 
        % Bootstrap CI methods: 1) percentile, 2) basic, 3) normal 
        ACFboot = NaN(N*nlags*M,B);
        COVboot = NaN(N*(N+1)/2*M,B);
        CORboot = NaN(N*(N-1)/2*M,B);
        PCboot = NaN(N*(N+1)/2*M,B);
        PCb = NaN(N,N,M);
        Aboot = parsboot.A;
        Cboot = parsboot.C;
        Qboot = parsboot.Q;
        Rboot = parsboot.R;
        Zboot = parsboot.Z;
        for b = 1:B
            test = Aboot(1,1,1,1,b);
            if isnan(test)
                continue 
            end
            parsb = struct('A',Aboot(:,:,:,:,b), 'C',Cboot(:,:,:,b), ...
                 'Q',Qboot(:,:,:,b), 'R',Rboot(:,:,b));
            [ACFb,~,COVb,VARb] = get_covariance(parsb,nlags,0);
            CORb = NaN(N,N,M);
            for j = 1:M
                Cj = Cboot(:,:,j,b);
                PCb(:,:,j) = Cj * ((Cj' * Cj) \ (Cj')); 
                SDj = sqrt(VARb(:,j));
                SDj(SDj < eps(1)) = 1;
                CORb(:,:,j) = (1 ./ SDj) .* COVb(:,:,j) ./ (SDj');
            end
            PCboot(:,b) = PCb(mask_COV);
            ACFboot(:,b) = ACFb(mask_ACF);
            COVboot(:,b) = COVb(mask_COV);
            CORboot(:,b) = CORb(mask_COR); 
        end
        ACFboot = reshape(ACFboot,[],M,B);
        COVboot = reshape(COVboot,[],M,B);
        CORboot = reshape(CORboot,[],M,B);
        PCboot = reshape(PCboot,[],M,B);
        Rboot = reshape(Rboot,N^2,B);
        Rboot = Rboot(mask_R,:);    

        % Match bootstrap estimates to MLE based on stationary covariances
        if M > 1
            d = zeros(factM,B);
            for m = 1:factM
                e = COVhat_tmp - COVboot(:,sigma(m,:),:);
                d(m,:) = squeeze(sum(abs(e),1:2));
            end
            [~,idx] = min(d); % best permutations
            for m = 1:factM
                if isequal(sigma(m,:),1:M)
                    continue
                end
                idx2 = find(idx == m); 
                % If best permutation is not identity matrix, apply it to
                % bootstrap parameter estimates
                if ~isempty(idx2) 
                    ACFboot(:,:,idx2) = ACFboot(:,sigma(m,:),idx2);
                    COVboot(:,:,idx2) = COVboot(:,sigma(m,:),idx2);
                    CORboot(:,:,idx2) = CORboot(:,sigma(m,:),idx2);
                    PCboot(:,:,idx2) = PCboot(:,sigma(m,:),idx2);
                    Zboot(:,:,idx2) = Zboot(sigma(m,:),sigma(m,:),idx2);                
                end
            end
        end
        COVhat_tmp = COVhat_tmp(:);

        % Reshape bootstrap estimates
        ACFboot = reshape(ACFboot,[],B);
        COVboot = reshape(COVboot,[],B);
        CORboot = reshape(CORboot,[],B);
        PCboot = reshape(PCboot,[],B);
        Zboot = reshape(Zboot,M^2,B);



%-------------------------------------------------------------------------%
%                     Bootstrap: Pointwise inference                      %
%-------------------------------------------------------------------------%

            mle = {COVhat_tmp,CORhat_tmp,ACFhat_tmp,PChat,Zhat_tmp,Rhat_tmp};
            boot = {COVboot,CORboot,ACFboot,PCboot,Zboot,Rboot};
            coverage_tmp = NaN(nboot_ci,npars_ci);

            for k = 1:npars_ci
                % Summary statistics
                mean_boot = mean(boot{k},2,'omitNaN');
                sd_boot = std(boot{k},1,2,'omitNaN');
                qt1_boot = quantile(boot{k},alpha/2,2);
                qt2_boot = quantile(boot{k},1-alpha/2,2);
                switch k
                    case 1
                        summary_boot_COV(:,:,rep) = ...
                            [mean_boot, sd_boot, qt1_boot, qt2_boot];
                    case 2
                        summary_boot_COR(:,:,rep) = ...
                            [mean_boot, sd_boot, qt1_boot, qt2_boot];
                    case 3
                        summary_boot_ACF(:,:,rep) = ...
                            [mean_boot, sd_boot, qt1_boot, qt2_boot];
                    case 4
                        summary_boot_PC(:,:,rep) = ...
                            [mean_boot, sd_boot, qt1_boot, qt2_boot];
                    case 5
                        summary_boot_Z(:,:,rep) = ...
                            [mean_boot, sd_boot, qt1_boot, qt2_boot];
                    case 6
                        summary_boot_R(:,:,rep) = ...
                            [mean_boot, sd_boot, qt1_boot, qt2_boot];
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
                "VariableNames",["COV","COR","ACF","PC","Z","R"]);
            disp(tmp);


            if mod(rep,10) == 0
                save(outfile);
            end
    end


    clear  mean_boot sd_boot qt1_boot qt2_boot boot
    clear parsboot Aboot Cboot Qboot Rboot Zboot ACFb COVb CORb parsb Cb PCb
    clear d e idx idx2 factM j sigma_best tmp stable
    clear ACFhat_tmp COVhat_tmp CORhat_tmp coverage_tmp Rhat_tmp Zhat_tmp
    clear LL LL2 pars pars0 pars1 pars2 S S0  b test y
    clear mask_ACF mask_COV mask_COR mask_R
    clear COVj Abig eigA Chatj Cj Q_j S Shat Shat2 S_perm target mle 

    toc

 
%-------------------------------------------------------------------------%
%                             Save results                                %
%-------------------------------------------------------------------------%

    save(outfile)

    
end


%%

% Coverage of MLE
% bias = mean(Ahat,2) - A;
% sd = std(Ahat,0,2);
% coverage_A = mean(abs(Ahat - bias - A) <= q_alpha * sd, 'all')
% 
% bias = mean(Rhat,2) - R;
% sd = std(Rhat,0,2);
% coverage_R = mean(abs(Rhat - bias - R) <= q_alpha * sd, 'all')
% 
% bias = mean(Zhat,2) - Z;
% sd = std(Zhat,0,2);
% coverage_Z = mean(abs(Zhat - bias - Z) <= q_alpha * sd, 'all')
% 
% bias = mean(ACFhat,2) - ACF;
% sd = std(ACFhat,0,2);
% coverage_ACF = mean(abs(ACFhat - bias - ACF) <= q_alpha * sd, 'all')
% 
% bias = mean(COVhat,2) - COV(:);
% sd = std(COVhat,0,2);
% coverage_COV = mean(abs(COVhat - bias - COV(:)) <= q_alpha * sd, 'all')
% 



 


