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
nrep = 2; % 00; % number of replications for each (N,T)
B = 100; % number of bootstrap replicates
nlags = 5; % number of lags at which to evaluate estimation
parallel = true;
conf_level = 0.9;
trgt_list = {'COV','COR','ACF','PC','Z','R'}; % target parameters
ci_method = {'percentile','basic','normal'}; 
% rng(14) % set seed for reproducibility

% EM control parameters
control = struct('eps',1e-6,'ItrNo',50,'beta0',1,'betarate',1,...
    'safe',false,'verbose',false);
control_boot = struct('eps',1e-6,'ItrNo',50,'verbose',false);
control2 = control; control2.ItrNo = 500;
fixed = [];  
equal = struct('mu',true,'Sigma',true);
scale = struct('A',.99);  
opts = struct('segmentation','fixed','len',50,'Replicates',50);



%-------------------------------------------------------------------------%
%            PRIMARY SIMULATION LOOP OVER DATA DIMENSIONS N,T             %
%-------------------------------------------------------------------------%

warning("off")
npars_ci = numel(trgt_list); 
nboot_ci = numel(ci_method); 

for i = 1:nN*nT

    tic
    N = NTgrid(1,i);
    T = NTgrid(2,i); 


    fprintf('Simulations for N=%d T=%d\n',N,T);
   
    % Output structures
    coverage = NaN(nboot_ci,npars_ci,nrep);
    outfile = sprintf('result_sim_obs_boot_N%dT%d.mat',N,T);


    
    
%-------------------------------------------------------------------------%
%                         LOOP OVER REPLICATIONS                          %
%-------------------------------------------------------------------------%

  

    for rep = 1:nrep

        fprintf("Replication %d\n",rep)




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
            sig2Q = .005;
            Q = zeros(r,r,M);
            for j = 1:M
                Q_j = randn(r);
                Q(:,:,j) = sig2Q * (Q_j' * Q_j);
            end

            % Observation noise covariance matrix
            sig2R = .005/N;
            rhoR = 0.1; % exchangeable noise structure
            R = rhoR * sig2R * ones(N) + sig2R * (1-rhoR) * eye(N);

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
        mask = struct();
        mask.COV = logical(repmat(tril(ones(N)),[1,1,M]));
        mask.COR = logical(repmat(tril(ones(N),-1),[1,1,M]));
        mask.PC = find(mask.COV);
        mask.ACF = true(N,nlags+1,M);
        mask.ACF(:,1,:) = false;
        mask.Z = true(M,M);
        mask.R = mask.COV(:,:,1);

        % Reshape model parameters
        COV = COV(mask.COV);
        COR = stationary.COR(mask.COR);
        ACF = stationary.ACF(mask.ACF);
        PC = zeros(N,N,M);
        for j = 1:M
            Cj = C(:,:,j);
            PC(:,:,j) = Cj*((Cj'*Cj)\(Cj'));
        end
        PC = PC(mask.PC);
        R = R(mask.R);
        Z = Z(mask.Z);

        % Inference targets
        target = struct('COV',COV, 'COR',COR, 'ACF',ACF, 'PC',PC, ...
            'Z',Z, 'R',R);




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
            % Permutations for matching
            sigma = perms(1:M); 
            factM = factorial(M);
            classif = NaN(factM,1);
            for m = 1:factM
                S_perm = sigma(m,Shat);                           
                classif(m) = mean(S_perm == S);
            end
            [~,idx] = max(classif);
            sigma_best = sigma(idx,:);

            % Re-arrange estimated regimes and parameters as needed
            if ~isequal(sigma_best,1:M)
                Shat = sigma_best(Shat);
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
            Cj = pars.C(:,:,j);
            PChat(:,:,j) = Cj * ((Cj'*Cj)\(Cj')); 
        end
        PChat = PChat(:);

    





%-------------------------------------------------------------------------%
%                               Bootstrap                                 %
%-------------------------------------------------------------------------%


           % Parametric bootstrap
        match = 'COV';
        if M == 2
            match = 'no';
        end
        [parsboot,LLboot] = bootstrap_obs(pars,T,B,opts,...
                control_boot,equal,fixed,scale,parallel,match);

        % Matching based on initial regime (only works for M=2)
        if M == 2
            [~,Sb1] = max(parsboot.Pi);
            test = (Sb1 ~= Shat(1));
            if any(test)
                rev = flip(1:M);
                parsboot.A(:,:,:,:,test) = parsboot.A(:,:,:,rev,test); 
                parsboot.C(:,:,:,test) = parsboot.C(:,:,rev,test); 
                parsboot.Q(:,:,:,test) = parsboot.Q(:,:,rev,test);                
                parsboot.Z(:,:,test) = parsboot.Z(rev,rev,test);
            end
        end
            
       % Bootstrap distribution of projection matrix associated with C
        PCboot = NaN(N,N,M,B);
        for b = 1:B
            for j = 1:M
                Cb = parsboot.C(:,:,j,b);  
                PCboot(:,:,j,b) = Cb * ((Cb' * Cb) \ (Cb'));
            end
        end
        PCboot = reshape(PCboot,M*N^2,B);


        
%-------------------------------------------------------------------------%
%                     Bootstrap: Pointwise inference                      %
%-------------------------------------------------------------------------%



        % Calculate bootstrap confidence intervals
        ci = bootstrap_ci(parsboot,pars,conf_level,nlags);
 
        % Add CI for projection matrix associated with C
        alpha = 1 - conf_level;
        z = norminv(1-alpha/2); 
        mean_boot = mean(PCboot,2,'omitNaN');
        sd_boot = std(PCboot,1,2,'omitNaN');
        qt1_boot = quantile(PCboot,alpha/2,2);
        qt2_boot = quantile(PCboot,1-alpha/2,2);
        ci_PC = struct('percentile',[], 'basic',[], 'normal',[]); 
        ci_PC.percentile.lo = qt1_boot; 
        ci_PC.percentile.up = qt2_boot;
        ci_PC.basic.lo = 2 * PChat - qt2_boot; 
        ci_PC.basic.up = 2 * PChat - qt1_boot; 
        ci_PC.normal.lo = 2 * PChat - mean_boot - z *  sd_boot;
        ci_PC.normal.up = 2 * PChat - mean_boot + z *  sd_boot;
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
       
        % Display current coverage
        coverage(:,:,rep) = coverage_tmp;
        fprintf("Current coverage level\n");
        tmp = array2table(mean(coverage(:,:,1:rep),3,'omitNaN'),...
            'RowNames',ci_method, 'VariableNames',trgt_list);
        disp(tmp);

        % Periodically clean up workspace and save results
        if mod(rep,10) == 0
            clear A* alpha b C Cb Chat ci ci_name ci_PC Cj classif COR* COV* ...
            coverage_tmp eigA idx j k l lb LL* m mask* mean_boot ...
            Ms* pars* PC* Q* qt* R* S S_perm sd* sigma_best Shat* stable ...
            stationary* target theta tmp trgt_name ub y z Z* ...
            noise signal snr test
            save(outfile);
        end
    end
    toc

    % Save results at end of loop
    clear rep
    save(outfile)
  
end
