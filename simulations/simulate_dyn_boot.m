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
nrep = 5; %200; % number of replications for each (N,T)
B = 100; % number of bootstrap replicates
nlags = 5; % number of lags at which to evaluate estimation
parallel = true;
conf_level = 0.9;
trgt_list = {'COV','COR','ACF','PC','Z','R'}; % target parameters
ci_method = {'percentile','basic','normal'}; 
% rng(14) % set seed for reproducibility

% EM control parameters
control = struct('eps',1e-6,'ItrNo',50,'safe',false,'verbose',false);
fixed = [];  
equal = struct('mu',true,'Sigma',true);
scale = struct('A',.99);  
opts = struct('segmentation','fixed','len',50,'Replicates',50);



%-------------------------------------------------------------------------%
%            PRIMARY SIMULATION LOOP OVER DATA DIMENSIONS N,T             %
%-------------------------------------------------------------------------%

% Misc
npars_ci = numel(trgt_list); 
nboot_ci = numel(ci_method); 
coverage = NaN(nboot_ci,npars_ci,nrep); % output array for CI coverage
alpha = 1 - conf_level;
z_alpha = norminv(1-alpha/2); % quantile for normal CI



for i = 1:1 % nN*nT

    tic
    N = NTgrid(1,i);
    T = NTgrid(2,i);

    fprintf('Simulations for N=%d T=%d\n',N,T);
    warning("off")

    % Output structures
    ACFhat = NaN(N*nlags*M,nrep);
    COVhat = NaN(N*(N+1)/2*M,nrep);
    CORhat = NaN(N*(N-1)/2*M,nrep);
    PChat = NaN(N*(N+1)/2,nrep);
    Zhat = NaN(M^2,nrep);
    Rhat = NaN(N*(N+1)/2,nrep);



    outfile = sprintf('result_sim_dyn_boot_N%dT%d.mat',N,T);




%-------------------------------------------------------------------------%
%                      SECONDARY LOOP OVER REPLICATIONS                   %
%-------------------------------------------------------------------------%


        % Masks 
        mask_ACF = true(N,nlags+1,M);
        mask_ACF(:,1,:) = false;
        mask_ACF = mask_ACF(:);
        mask_R = logical(tril(ones(N)));
        mask_COV = logical(repmat(tril(ones(N)),[1,1,M]));
        mask_COR = logical(repmat(tril(ones(N),-1),[1,1,M]));
        mask = struct('ACF',mask_ACF, 'COV',mask_COV, 'COR',mask_COR, ...
            'PC',true(N*(N+1)/2,1), 'Z',true(M^2,1), 'R',mask_R);

    for rep = 1:nrep

        fprintf("Replication %d\n",rep)


%-------------------------------------------------------------------------%
%                   Generate model parameters                             %
%-------------------------------------------------------------------------%

        % Observation matrix    
        [C,~,~] = svd(randn(N,r),'econ');

       % Observation noise covariance matrix
        sig2R = 5e-3 / N;
        rhoR = 0.1; % exchangeable noise structure
        R = rhoR * sig2R * ones(N) + sig2R * (1-rhoR) * eye(N);

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

            % State noise covariance matrix
            sig2Q = 5e-3;
            Q = zeros(r,r,M);
            for j = 1:M
                Q_j = randn(r);
                Q(:,:,j) = sig2Q * (Q_j' * Q_j);
            end

            % Gather model parameters
            theta = struct('A',A, 'C',C, 'Q',Q, 'R',R);

            % Stationary variance matrices and autocorrelation functions
            stationary = get_covariance(theta,nlags,0);
            
            % Test SNR
            signal = zeros(1,M); 
            COV = stationary.COV;
            for j = 1:M
                signal(j) = sum(diag(COV(:,:,j))) - sum(diag(R));
            end
            noise = sum(diag(R));
            snr = all(signal >= 5 * noise & signal <= 10 * noise);        

        end

        clear signal noise snr test

        % Initial means and variances
        theta.mu = zeros(r,M);
        theta.Sigma = repmat(0.1 * eye(r),1,1,M);
        % Initial regime probabilities
        theta.Pi = [1;0];
        % Transition probability matrix
        Z = [.98,.02;.02,.98];
        theta.Z = Z;

        % Reshape model parameters
        A = A(:);
        PC = C*((C'*C)\(C'));
        PC = PC(mask.R);
        R = R(mask.R);
        Z = Z(:);

        % Covariance-related quantities
        ACF = stationary.ACF(mask.ACF);
%         COR = NaN(N,N,M);
%         for j = 1:M
%             COR(:,:,j) = corrcov(COV(:,:,j)+COV(:,:,j)');
%         end
%         COV = COV(mask_COV);
%         COR = COR(mask_COR);
        COV = stationary.COV(mask.COV);
        COR = stationary.COV(mask.COR);
        

        % Inference targets
        target = struct('COV',COV, 'COR',COR, 'ACF',ACF, 'PC',PC,...
            'Z',Z, 'R',R);



        % Turn off warnings
        warning("off")

        % Permutations for matching
        sigma = perms(1:M); 
        factM = factorial(M);



%-------------------------------------------------------------------------%
%                           Generate data                                 %
%-------------------------------------------------------------------------%


        [y,S] = simulate_dyn(theta,T);



%-------------------------------------------------------------------------%
%                         Fit switching SSM                               %
%-------------------------------------------------------------------------%

        % Initialization
        pars0 = init_dyn(y,M,p,r,opts,control,equal,fixed,scale);

         % EM algorithm           
         try
            [~,Ms,~,Shat,~,~,pars,LL] = ... 
                switch_dyn(y,M,p,r,pars0,control,equal,fixed,scale);
         catch
            continue                    
         end
         % Acceleration
         try 
             [~,~,pars1] = ...
                 fast_dyn(y,M,p,r,Shat,pars,control,equal,fixed,scale);
             [~,Ms2,~,Shat2,~,~,pars2,LL2] = ... 
                switch_dyn(y,M,p,r,pars1,control,equal,fixed,scale);
             if max(LL2) > max(LL)
                 pars = pars2;
                 Shat = Shat2;
                 Ms = Ms2;
             end
         catch
         end

        % Match estimated regimes to true regimes based on classification
        % rate (formerly: stationary covariance)
        classif = NaN(factM,1);
        for m = 1:factM
            S_perm = sigma(m,Shat);                           
            classif(m) = mean(S_perm == S);
        end
        [~,idx] = max(classif);
        sigma_best =  sigma(idx,:);

        % Re-arrange estimated regimes and parameters as needed
        if ~isequal(sigma_best,1:M)
    %         Shat = sigma_best(Shat);
            pars.A(:,:,:,sigma_best) = pars.A;
            pars.Q(:,:,sigma_best) = pars.Q;
            pars.mu(:,sigma_best) = pars.mu;
            pars.Sigma(:,:,sigma_best) = pars.Sigma;
            pars.Pi(sigma_best) = pars.Pi;
            pars.Z(sigma_best,sigma_best) = pars.Z;
            Ms(sigma_best,:) = Ms;
        end

        % Projection matrix on state space
        Chat = pars.C;
        PChat_tmp = Chat * ((Chat'*Chat)\(Chat')); 
        PChat_tmp = PChat_tmp(mask.R);
        PChat(:,rep) = PChat_tmp;

        % Stationary, regime-specific covariance and autocorrelation 
        stationary_tmp = get_covariance(pars,nlags,0);
        ACFhat_tmp = stationary_tmp.ACF(mask.ACF);
        COVhat_tmp = stationary_tmp.COV(mask.COV);
        CORhat_tmp = stationary_tmp.COR(mask.COR);

        % Reshape MLE
        Rhat_tmp = pars.R(mask.R);
        Rhat(:,rep) = Rhat_tmp;
        Zhat_tmp = pars.Z(:);
        Zhat(:,rep) = Zhat_tmp;
        COVhat(:,rep) = COVhat_tmp;
        CORhat(:,rep) = CORhat_tmp;
        ACFhat(:,rep) = ACFhat_tmp;





%-------------------------------------------------------------------------%
%                               Bootstrap                                 %
%-------------------------------------------------------------------------%

    
        % Parametric bootstrap
        match = 'COV';
         [parsboot,LLboot] = bootstrap_dyn(pars,T,B,opts,...
                control,equal,fixed,scale,parallel,match);
    %         
        % Nonparametric bootstrap
    %     [parsboot,LLboot] = bootstrap_dyn_npar(pars,y,Ms,B,opts,...
    %             control,equal,fixed,scale,parallel);


        % Bootstrap distribution of projection matrix associated with C
        PCboot = NaN(N*(N+1)/2,B);
        for b = 1:B
            Cb = parsboot.C(:,:,b);
            PCb = Cb * ((Cb' * Cb) \ (Cb'));
            PCboot(:,b) = PCb(mask.R);
        end
    
       
    
   


    
%-------------------------------------------------------------------------%
%                     Bootstrap: Pointwise inference                      %
%-------------------------------------------------------------------------%

%         mle = {COVhat_tmp,CORhat_tmp,ACFhat_tmp,PChat_tmp,Zhat_tmp,Rhat_tmp};
%         boot = {COVboot,CORboot,ACFboot,PCboot,Zboot,Rboot};

        % Calculate bootstrap confidence intervals
        ci = bootstrap_ci(parsboot,pars,1-alpha,nlags);
        
        % Add CI for projection matrix associated with C
        mean_boot = mean(PCboot,2,'omitNaN');
        sd_boot = std(PCboot,1,2,'omitNaN');
        qt1_boot = quantile(PCboot,alpha/2,2);
        qt2_boot = quantile(PCboot,1-alpha/2,2);
        ci_PC = struct('percentile',[], 'basic',[], 'normal',[]); 
        ci_PC.percentile.lo = qt1_boot; 
        ci_PC.percentile.up = qt2_boot;
        ci_PC.basic.lo = 2 * PChat_tmp - qt2_boot; 
        ci_PC.basic.up = 2 * PChat_tmp - qt1_boot; 
        ci_PC.normal.lo = 2 * PChat_tmp - mean_boot - z_alpha *  sd_boot;
        ci_PC.normal.up = 2 * PChat_tmp - mean_boot + z_alpha *  sd_boot;
        ci.PC = ci_PC;
        
        coverage_tmp = NaN(nboot_ci,npars_ci);
        
        % Loop over target parameters
        for k = 1:npars_ci
            trgt_name = trgt_list{k};
            
            % Loop over CI methods
            for l = 1:nboot_ci
                ci_name = ci_method{l};
                
                % Remove redundant parameters from CIs 
                lb = ci.(trgt_name).(ci_name).lo(mask.(trgt_name));
                ub = ci.(trgt_name).(ci_name).up(mask.(trgt_name));
                
                % Calculate average coverage
                coverage_tmp(l,k) = mean(lb <= target.(trgt_name) & ...
                    target.(trgt_name) <= ub);
            end
        end
%             % Summary statistics
%             mean_boot = mean(boot{k},2,'omitNaN');
%             sd_boot = std(boot{k},1,2,'omitNaN');
%             qt1_boot = quantile(boot{k},alpha/2,2);
%             qt2_boot = quantile(boot{k},1-alpha/2,2);
%             switch k
%                 case 1
%                     summary_boot_COV(:,:,rep) = ...
%                         [mean_boot, sd_boot, qt1_boot, qt2_boot];
%                 case 2
%                     summary_boot_COR(:,:,rep) = ...
%                         [mean_boot, sd_boot, qt1_boot, qt2_boot];
%                 case 3
%                     summary_boot_ACF(:,:,rep) = ...
%                         [mean_boot, sd_boot, qt1_boot, qt2_boot];
%                 case 4
%                     summary_boot_PC(:,:,rep) = ...
%                         [mean_boot, sd_boot, qt1_boot, qt2_boot];
%                 case 5
%                     summary_boot_Z(:,:,rep) = ...
%                         [mean_boot, sd_boot, qt1_boot, qt2_boot];
%                 case 6
%                     summary_boot_R(:,:,rep) = ...
%                         [mean_boot, sd_boot, qt1_boot, qt2_boot];
%             end
%             
%             % Percentile bootstrap CI
%             coverage_tmp(1,k) = ...
%                 mean(qt1_boot <= target{k} & target{k} <= qt2_boot);
%             % Basic bootstrap CI
%             coverage_tmp(2,k) = mean(2 * mle{k} - qt2_boot <= target{k} & ...
%                 target{k} <= 2 * mle{k} - qt1_boot);
%             % Normal bootstrap CI
%             coverage_tmp(3,k) = mean(abs(2 * mle{k} - mean_boot - target{k})...
%                 <= z_alpha * sd_boot);
        
        
        coverage(:,:,rep) = coverage_tmp;
        fprintf("Current coverage level\n");
        tmp = array2table(mean(coverage(:,:,1:rep),3,'omitNaN'),...
            'RowNames',ci_method, 'VariableNames',trgt_list);
        disp(tmp);


        if mod(rep,10) == 0
            save(outfile);
        end
    end



    
%     clear mask_ACF mask_COV mask_COR mask_R
%     clear  mean_boot sd_boot qt1_boot qt2_boot boot
%     clear parsboot Aboot Cboot Qboot Rboot Zboot ACFb COVb CORb parsb Cb PCb
%     clear ci ci_PC classif COV COVhat COR CORhat i k l lb 
%     clear d e idx idx2 factM j sigma_best tmp stable
%     clear ACFhat_tmp COVhat_tmp CORhat_tmp coverage_tmp Rhat_tmp Zhat_tmp PChat_tmp
%     clear LL LL2 pars pars0 pars1 pars2 S S0  b test y
%     clear COVj Abig eigA Chatj Cj Q_j S Shat Shat2 S_perm target mle 

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



 


  