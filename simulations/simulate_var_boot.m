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
trgt_list = {'A','Q','Z','COV','COR','ACF'}; % target parameters
ci_method = {'percentile','basic','normal'}; 

% EM control parameters
control = struct('eps',1e-6,'ItrNo',200,'verbose',false);
control2 = control; control2.ItrNo = 800;
control_boot = struct('eps',1e-6,'ItrNo',100,'verbose',false);
equal = struct('mu',true,'Sigma',true);
scale = struct('A',.999);  
opts = struct('segmentation','fixed','len',50,'Replicates',50);

% rng(14) % set seed for reproducibility




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
    outfile = sprintf('result_sim_var_boot_N%dT%d.mat',N,T);

    fixed = struct('A',repmat(diag(NaN(N,1)),[1,1,p,M]));  


    
%-------------------------------------------------------------------------%
%                   SECONDARY LOOP OVER REPLICATIONS                      %
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
            % Transition matrices for dynamics
            A = zeros(N,N,p,M);
            for j = 1:M
                A(:,:,1,j) = diag(rand(N,1) * .1 + .85);
                A(:,:,2,j) = diag(rand(N,1) * .1 - .05);
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
            sig2Q = 0.01/N;
            Q = zeros(N,N,M);
            for j = 1:M
                Q_j = randn(N);
                Q(:,:,j) = sig2Q * (Q_j' * Q_j);
            end

            % Stationary variance matrices and autocorrelation functions
            theta = struct('A',A,'Q',Q);
            stationary = get_covariance(theta,nlags,0);
            COV = stationary.COV;

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

        % Masks 
        mask = struct('A',true(size(A)), 'Z',true(size(theta.Z)));
        mask.ACF = true(N,nlags+1,M);
        mask.ACF(:,1,:) = false;
        mask.COV = logical(repmat(tril(ones(N)),[1,1,M]));
        mask.COR = logical(repmat(tril(ones(N),-1),[1,1,M]));
        mask.Q = mask.COV;
        
        % Reshape model parameters 
        A = A(:);
        Q = Q(mask.Q);
        Z = theta.Z(:);
        ACF = stationary.ACF(mask.ACF);
        COV = COV(mask.COV);
        COR = stationary.COR(mask.COR);
        target = struct('A',A, 'Q',Q, 'Z',Z, 'ACF',ACF, 'COV',COV, ...
            'COR',COR);
        
        % Permutations for matching
        sigma = perms(1:M); 
        factM = factorial(M);



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
            [~,Ms,~,Shat,pars,LL] = ... 
                switch_var(y,M,p,pars0,control,equal,fixed,scale);
         catch
            continue                    
         end

        % Acceleration
        try 
             pars1 = fast_var(y,M,p,Shat,control2,equal,fixed,scale);
             [~,Ms2,~,Shat2,pars2,LL2] = ... 
                switch_var(y,M,p,pars1,control,equal,fixed,scale);
             if max(LL2) > max(LL)
                 pars = pars2;
                 Shat = Shat2;
                 Ms = Ms2;
             end
        catch
        end

        % Match estimates to model parameters by regime
        classif = zeros(1,factM);    
        for m = 1:factM
            S_perm = sigma(m,Shat);                           
            classif(m) = mean(S_perm == S);
        end
        [~,idx] = max(classif);
        sigma_best = sigma(idx,:);

       % Re-arrange estimated regimes and parameters as needed
        if ~isequal(sigma_best,1:M)
            Shat = sigma_best(Shat);
%             pars.A = pars.A(:,:,:,sigma_best);
%             pars.Q = pars.Q(:,:,sigma_best);
%             pars.mu = pars.mu(:,sigma_best);
%             pars.Sigma = pars.Sigma(:,:,sigma_best);
%             pars.Pi = pars.Pi(sigma_best);
%             pars.Z = pars.Z(sigma_best,sigma_best);
            pars.A(:,:,:,sigma_best) = pars.A;
            pars.Q(:,:,sigma_best) = pars.Q;
            pars.mu(:,sigma_best) = pars.mu;
            pars.Sigma(:,:,sigma_best) = pars.Sigma;
            pars.Pi(sigma_best) = pars.Pi;
            pars.Z(sigma_best,sigma_best) = pars.Z;
        end





%-------------------------------------------------------------------------%
%                               Bootstrap                                 %
%-------------------------------------------------------------------------%


        % Parametric bootstrap
        match = 'COV';
        if M == 2
            match = 'no';
        end
        [parsboot,LLboot] = bootstrap_var(pars,T,B,opts,...
                control_boot,equal,fixed,scale,parallel,match);

        % Nonparametric boostrap
    %     [parsboot,LLboot] = bootstrap_var_npar(pars,y,S,B,opts,...
    %         control_boot,equal,fixed,scale,parallel);


        % Matching based on initial regime (only works for M=2)
        if M == 2
            [~,Sb1] = max(parsboot.Pi);
            test = (Sb1 ~= Shat(1));
            if any(test)
                rev = flip(1:M);
                parsboot.A(:,:,:,:,test) = parsboot.A(:,:,:,rev,test); 
                parsboot.Q(:,:,:,test) = parsboot.Q(:,:,rev,test);                
                parsboot.Z(:,:,test) = parsboot.Z(rev,rev,test);
            end
        end


%-------------------------------------------------------------------------%
%                     Bootstrap: Pointwise inference                      %
%-------------------------------------------------------------------------%


        % Calculate bootstrap confidence intervals
        ci = bootstrap_ci(parsboot,pars,conf_level,nlags);
    
        coverage_tmp = NaN(nboot_ci,npars_ci);
        
        % Loop over taget parameters
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

        % Clean up and save results periodically 
         
        if mod(rep,10) == 0
            clear A* ci ci_name classif COR COV coverage_tmp eigA factM ...
                idx j k l lb LL* m Ms* mask* noise pars* Q Q_j rev S* ...
                signal sigma_best snr stable stationary target test ...
                theta tmp trgt_name ub y Z
            save(outfile)
        end
    
    end    
    toc
    
    % Save results at end of loop
    clear rep
    save(outfile)
    
end


