%==========================================================================
%  Simulation study of switching vector autoregressive model 
%==========================================================================


clc; clearvars; close all;

% Simulation parameters
% Ngrid = [10,50,100]; % time series dimension (# variables) 
% Tgrid = [400,600,800,1000]; % time series length
Ngrid = 10; Tgrid = 600;
nN = numel(Ngrid);
nT = numel(Tgrid);
NTgrid = [repelem(Ngrid,1,nT); repmat(Tgrid,1,nN)];
nrep = 500; % number of replications for each N

% Result arrays
method_list = {'sw-km', 'switch-ols', 'switch-ml', 'or'};
nmethods = numel(method_list);
% Note: OR-OLS and OR-ML are identical in switching VAR
npars = 3; % A Q Z
nlags = 5; % number of lags at which to evaluate estimation
len = 31; % window length for SW-KM

% Data and model dimensions
M = 2; % number of regimes
p = 2; % VAR order

for i = 1:nN*nT
   
    tic
    N = NTgrid(1,i);
    T = NTgrid(2,i);
    fprintf('Simulations for N=%d T=%d\n',N,T);
    
    %@@@@@ SEGMENTATION OF TIME SERIES IN REGIMES
    % Methods compared: sliding windows + K-means (1), initial estimates of
    % switching SSM (2) and final estimates of switching SSM (3)
    % Performance measures: Rand index, number of detected change points, 
    % maximum distance between detected change point and true set of change
    % points {100,200,300}
    % Index 1: replication index
    % Index 2: method (1=sliding, 2=initial ssm, 3=final ssm, 4=oracle)
    % Index 3: index in Ngrid
    CLASSIF = NaN(nmethods,nrep);  % Classification rate
    HAUSDORFF = NaN(nmethods,nrep);   % Hausdorff distance between estimated and true change points
    RAND = NaN(nmethods,nrep);  % Rand index
    NCP = NaN(nmethods,nrep);   % number of detected change points

    %@@@@@ COVARIANCE ESTIMATION (STEADY-STATE CONNECTIVITY)
    % Methods compared: sliding windows + K-means (1), final estimates of
    % switching SSM (2), OLS estimate with oracle knowledge of regimes (3), ML
    % estimate with oracle knowledge of regimes (4) 
    % Performance measures: relative error and mean squared error in estimation 
    % of V(y(t)|S(t)=j) and V(Cx(t)|S(t)=j). For target V(j) and estimate Vhat(j), 
    % RE = || Vhat(:)-V(:) ||_F / || V(:) ||_F 
    % MSE = || Vhat(:)-V(:) ||^2_F / size(V) 
    % Note: V(Cx(t)|S(t)=j) cannot be estimated by method (1) --> skipped
    % with || ||_F Frobenius norm
    % Index 1: method (1=sliding, 2=initial ssm, 3=final ssm, 4=oracle)
    % Index 2: lag (1 = lag 0, 2 = lag 1, ...) 
    % Index 3: replication index
    RE_COV = NaN(nmethods,nrep);
    MSE_COV = NaN(nmethods,nrep);
    RE_COR = NaN(nmethods,nrep);
    MSE_COR = NaN(nmethods,nrep);
    RE_ACF = NaN(nmethods,nrep);
    MSE_ACF = NaN(nmethods,nrep);


    %@@@@@ PARAMETER ESTIMATION 
    % Performance measures: relative error and mean squared error (same as in 
    % covariance estimation)
    % Index 1: target (1=A, 2=Q, 3=Z)
    % Index 2: method (1=sliding (N/A), 2=initial ssm, 3=final ssm, 4=oracle)
    % Index 3: replication index
    RE_theta = NaN(nmethods,npars,nrep);
    MSE_theta = NaN(nmethods,npars,nrep);


    % Masks
    mask_ACF = true(N,nlags+1,M);
    mask_ACF(:,1,:) = false;
    mask_ACF = find(mask_ACF);
    mask_COV = logical(repmat(tril(ones(N)),[1,1,M]));
    mask_COV = find(mask_COV);
    mask_COR = logical(repmat(tril(ones(N),-1),[1,1,M]));
    mask_COR = find(mask_COR);

    warning('off')
    
    parfor_progress(nrep);

    parfor rep = 1:nrep

        warning('off')
  

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
            sig2Q = 1e-2/N;
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
                signal(j) = sum(diag(COV(:,:,j))) - sum(diag(Q(:,:,j)));
                noise(j) = sum(diag(Q(:,:,j)));
            end
            snr = all(signal >= 5 * noise & signal <= 10 * noise);        

        end

        % Initial means and variances
        mu = zeros(N,M);
        Sigma = repmat(0.1 * eye(N),1,1,M);
        % Initial regime probabilities
        Pi = [1;0];
        % Transition probability matrix
        Z = [.98,.02;.02,.98];

        % Gather all parameters in one structure
        theta.mu = mu;
        theta.Sigma = Sigma;
        theta.Pi = Pi; 
        theta.Z = Z;

         % reshape model parameters
        A = A(:);
        Q = Q(mask_COV);
        Z = Z(:);
        ACF = ACF(mask_ACF);
        COR = zeros(N,N,M);
        for j = 1:M
            COR(:,:,j) = corrcov(COV(:,:,j) + COV(:,:,j)');
        end
        COV = COV(mask_COV);
        COR = COR(mask_COR);
       
        
%-------------------------------------------------------------------------%
%                    Simulate regimes and observations                    %
%-------------------------------------------------------------------------%

        [y,S] = simulate_var(theta,T);
        cp = find(diff(S) ~= 0) + 1;
        
    
        
%-------------------------------------------------------------------------%
%                   Parameters for switching SSM model fit                %
%-------------------------------------------------------------------------%
        
        
        % Control parameters
        control = struct('eps',1e-6,'ItrNo',100,'safe',false,'verbose',false);
        control2 = control; control2.ItrNo = 200;
        opts = struct('segmentation','fixed','len',50, 'tol',.05, ...
            'Replicates',100);

        % Fixed coefficients: use "one in ten" rule. 
        if T >= 10*M*p*N 
            fixed = []; 
        else
            fixed = struct('A',repmat(diag(NaN(N,1)),[1,1,p,M]));
        end

        % Equality constraints 
        equal = struct('mu',true,'Sigma',true);

        % Scale of (columns of) matrices C(j) and/or upper bound for eigenvalues
        % of A(j)
        scale = struct('A',.99); 
         
        
        %@@@@@ LOOP OVER ESTIMATORS
        % 1= sliding window + K-means, 2=switching SSM (OLS), 
        % 3 = switching SSM (ML), 4 = oracle SSM (OLS), 5 = oracle SSM (ML)
        
        CLASSIF_tmp = NaN(nmethods,1);
        NCP_tmp = NaN(nmethods,1);
        HAUSDORFF_tmp = NaN(nmethods,1);
        RAND_tmp = NaN(nmethods,1);
        RE_COV_tmp = NaN(nmethods,1);
        MSE_COV_tmp = NaN(nmethods,1);
        RE_COR_tmp = NaN(nmethods,1);
        MSE_COR_tmp = NaN(nmethods,1);
        RE_ACF_tmp = NaN(nmethods,1);
        MSE_ACF_tmp = NaN(nmethods,1);
        RE_theta_tmp = NaN(nmethods,npars);
        MSE_theta_tmp = NaN(nmethods,npars);
        method_list_tmp = method_list;

        for k = 1:nmethods 
            
            pars = []; Shat = [];      
            method_name = method_list_tmp{k};
            switch method_name
            
                case 'sw-km'
        
%-------------------------------------------------------------------------%
%               Sliding windows covariance estimation                     %
%-------------------------------------------------------------------------%


            if N == 10
                nrepkm = 100;
            else
                nrepkm = 50;
            end
            [Shat,Zhat] = swkm(y,M,len,nrepkm);

            
                    
%-------------------------------------------------------------------------%
%                         Fit switching SSM                               %
%-------------------------------------------------------------------------%
 
                case 'switch-ols'
                    
                    % Initialization (Switching VAR - OLS)
                    [pars0,S0] = ...
                        init_var(y,M,p,opts,control,equal,fixed,scale);
                    pars = pars0; Shat = S0;
                    
                case 'switch-ml' 
                     try
                        [~,~,~,Shat,pars,LL] = ... 
                            switch_var(y,M,p,pars0,control,equal,fixed,scale);
                     catch
                        continue                    
                     end
                     try 
                         pars1 = ...
                             fast_var(y,M,p,Shat,control2,equal,fixed,scale);
                         [~,~,~,Shat2,pars2,LL2] = ... 
                            switch_var(y,M,p,pars1,control,equal,fixed,scale);
                         if max(LL2) > max(LL)
                             Shat = Shat2; pars = pars2;
                         end
                     catch
                         continue
                     end
                                                            
                case 'or'
                    try 
                        % Oracle estimator (OLS and ML are identical) 
                        pars = fast_var(y,M,p,S,control,equal,fixed,scale);
%                         pars.Z = theta.Z; 
                        Shat = S;
                    catch
                        continue
                    end
            end
            
            
            
%-------------------------------------------------------------------------%
%                       Classification performance                        %
%-------------------------------------------------------------------------%

  
            % Rand index
            RAND_tmp(k) = rand_index(Shat,S);

            % Detected change points
            cphat = find(diff(Shat) ~= 0) + 1;

            % Number of detected change-points
            NCP_tmp(k) = numel(cphat);

            % Hausdorff distance between estimated and true change points
            if isempty(cphat) 
                HAUSDORFF_tmp(k) = Inf;
            else
                HAUSDORFF_tmp(k) = max(... % fixed *after* running round 4
                    max(arrayfun(@(x) min(abs(x-cp)),cphat)), ...
                    max(arrayfun(@(x) min(abs(x-cphat)),cp)));
            end              

             % Classification rate
            sigma = perms(1:M); 
            factM = size(sigma,1); 
            classif = zeros(factM,1);
            for m = 1:factM
                S_perm = sigma(m,Shat);                           
                classif(m) = mean(S_perm == S);
            end
            [classif_max,idx] = max(classif);
            CLASSIF_tmp(k) = classif_max;
            sigma = sigma(idx,:);
                        
            % Re-arrange estimated regimes and parameters as needed
            if ~isequal(sigma,1:M)
                Shat = sigma(Shat);
                if strcmp(method_name,'sw-km')
                    Zhat(sigma,sigma) = Zhat; 
                else    
                    pars.A(:,:,:,sigma) = pars.A;
                    pars.Q(:,:,sigma) = pars.Q;
                    pars.Z(sigma,sigma) = pars.Z;
                    Zhat = pars.Z;
                 end
            end
              
       
            
            
%-------------------------------------------------------------------------%
%     Performance in cross-covariance estimation (connectivity measures)  %
%-------------------------------------------------------------------------%
      
            % Estimate long-run variance matrices V(y(t)|S(t)=j)
            if strcmp(method_name,'sw-km')
                COVhat = zeros(N,N,M);
                CORhat = zeros(N,N,M);
                ACFhat = zeros(N,N,nlags+1,M);
                burnin = 10; % drop first observations of each segment to reduce transients
                for j = 1:M
                    for l = 0:nlags
                        Sj = find(Shat == j);
                        Sj = Sj(Sj <= T-l);
                        idx = Sj(Shat(Sj+l) == j); % {t: S(t)=S(t+l)=j}
                        idx2 = idx(idx > burnin);
                        idx2 = idx2(Shat(idx2-burnin) == j); % {t: S(t-burnin)=S(t)=S(t+l)=j}
                        % If too few time points for accurate estimation, no burn-in 
                        if numel(idx2) < 10 
                            idx2 = idx;
                            if isempty(idx)
                                continue
                            end
                        end
                        m1 = mean(y(:,idx2),2);
                       s1 = std(y(:,idx2),1,2);
                    
                        n = numel(idx2);
                        if l == 0
                            COVhat(:,:,j) = (y(:,idx2)-m1) * (y(:,idx2)-m1)' / n;                            
                            s1(s1 < eps(1)) = 1;
                            CORhat(:,:,j) = (1 ./ s1) .* COVhat(:,:,j) ./ (s1');
                            ACFhat(:,1,j) = 1;
                        else
                            m2 = mean(y(:,idx2+l),2);
                            s2 = std(y(:,idx2+l),1,2);
                            s2(s2 < eps(1)) = 1;
                            CCVhat = (y(:,idx2+l)-m2) * (y(:,idx2)-m1)' / n; 
                            ACFhat(:,l+1,j) = diag(CCVhat) ./ (s1 .* s2);
                        end
                    end
                end
            else
                [ACFhat,~,COVhat,VARhat] = get_covariance(pars,nlags,0);
                CORhat = zeros(N,N,M);
                for j = 1:M
                    SDj = sqrt(VARhat(:,j));
                    SDj(SDj < eps(1)) = 1;
                    CORhat(:,:,j) = (1 ./ SDj) .* COVhat(:,:,j) ./ (SDj');
                end
            end
             
            % Reshape estimates
            COVhat = COVhat(mask_COV);
            CORhat = CORhat(mask_COR);
            ACFhat = ACFhat(mask_ACF);
            
            % Estimation performance
            err = COVhat - COV;
            RE_COV_tmp(k) =  norm(err,1) / norm(COV,1);
            MSE_COV_tmp(k) = mean(err.^2);
            err = CORhat - COR;
            RE_COR_tmp(k) =  norm(err,1) / norm(COR,1);
            MSE_COR_tmp(k) = mean(err.^2);
            err = ACFhat - ACF;
            RE_ACF_tmp(k) =  norm(err,1) / norm(ACF,1);
            MSE_ACF_tmp(k) = mean((err.^2));                 
      
      
            
%-------------------------------------------------------------------------%
%            Performance in estimation of parameters A,C,Q,R,Z            %
%-------------------------------------------------------------------------%

       
            if ~strcmp(method_name,'sw-km')
                % Reshape estimates
                Ahat = pars.A(:);
                Qhat = pars.Q(mask_COV);              
                
                RE_theta_tmp(k,1) = norm(Ahat-A,1) / norm(A,1);
                MSE_theta_tmp(k,1) = mean((Ahat-A).^2);
                RE_theta_tmp(k,2) = norm(Qhat-Q,1)/norm(Q,1);
                MSE_theta_tmp(k,2) = mean((Qhat-Q).^2);
            end
            Zhat = Zhat(:);
            RE_theta_tmp(k,3) = norm(Zhat-Z,1) / norm(Z,1);
            MSE_theta_tmp(k,3) = mean((Zhat-Z).^2);
        end            
        
        CLASSIF(:,rep) = CLASSIF_tmp;
        HAUSDORFF(:,rep) = HAUSDORFF_tmp;
        NCP(:,rep) = NCP_tmp;
        RAND(:,rep) = RAND_tmp;
        RE_COV(:,rep) = RE_COV_tmp;
        MSE_COV(:,rep) = MSE_COV_tmp;
        RE_COR(:,rep) = RE_COR_tmp;
        MSE_COR(:,rep) = MSE_COR_tmp;
        RE_ACF(:,rep) = RE_ACF_tmp;
        MSE_ACF(:,rep) = MSE_ACF_tmp;
        RE_theta(:,:,rep) = RE_theta_tmp;
        MSE_theta(:,:,rep) = MSE_theta_tmp;
        parfor_progress; 
    end

    parfor_progress(0); 
    toc
    

    % Duplicate the results for OR estimator (-> OR-OLS = OR-ML in VAR)
    idx = [1:nmethods,nmethods];
    CLASSIF = CLASSIF(idx,:); 
    HAUSDORFF = HAUSDORFF(idx,:);
    NCP = NCP(idx,:); 
    RAND = RAND(idx,:);
    RE_COV = RE_COV(idx,:); 
    MSE_COV = MSE_COV(idx,:);
    RE_COR = RE_COR(idx,:); 
    MSE_COR = MSE_COR(idx,:);
    RE_ACF = RE_ACF(idx,:); 
    MSE_ACF = MSE_ACF(idx,:);
    RE_theta = RE_theta(idx,:,:); 
    MSE_theta = MSE_theta(idx,:,:);
    
    
    
%-------------------------------------------------------------------------%
%                             Save results                                %
%-------------------------------------------------------------------------%

    clear -regexp mask
    clear idx ans

    outfile = sprintf('result_sim_var_N%dT%d.mat',N,T);
    % outfile = sprintf('result_swkm31_var_N%dT%d.mat',N,T);
    save(outfile);



    
end


