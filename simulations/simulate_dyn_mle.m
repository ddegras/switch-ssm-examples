%==========================================================================
%  Simulation study of state-space model with switching dynamics 
%==========================================================================


clc; clearvars; close all;

% Simulation parameters
Ngrid = [10,50,100]; % time series dimension (# variables) 
Tgrid = [400,600,800,1000]; % time series length
nN = numel(Ngrid);
nT = numel(Tgrid);
NTgrid = [repelem(Ngrid,1,nT); repmat(Tgrid,1,nN)];
nrep = 500; % number of replications for each N
method_list = {'sw-km', 'switch-ols', 'switch-ml', 'or-ols', 'or-ml'};
nmethods = numel(method_list);
npars = 5; % A C Q R Z
nlags = 5; % numer of lags at which to evaluate estimation
len = 31; % window length for SW-KM

% Data and model dimensions
M = 2; % number of regimes
p = 2; % VAR order
r = 2;  % state vector dimension


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
    % Index 2: method (1=sliding, 2=initial ssm, 3=final ssm, 4=oracle OLS, 5=oracle ML)
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
    % Index 1: method (1=sliding, 2=initial ssm, 3=final ssm, 4=oracle OLS, 5=oracle ML)
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
    % Index 1: target (1=A, 2=C, 3=Q, 4=R, 5=Z)
    % Index 2: method (1=sliding (N/A), 2=initial ssm, 3=final ssm, 4=oracle OLS, 5=oracle ML)
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
    mask_Q = logical(repmat(tril(ones(r)),[1,1,M]));
    mask_Q = find(mask_Q);
    mask_R = logical(tril(ones(N)));
    mask_R = find(mask_R);

    parfor_progress(nrep);

    parfor rep = 1:nrep

        warning('off')

        % Observation noise covariance
        sig2R = 0.005/N;
        rhoR = 0.1; % exchangeable noise structure
        R = rhoR * sig2R * ones(N) + sig2R * (1-rhoR) * eye(N);
        
        % Initial means and variances
        mu = zeros(r,M);
        Sigma = repmat(0.1 * eye(r),1,1,M);

        % Initial regime probabilities
        Pi = [1;0];
        
        % Transition probability matrix
        Z = [.98,.02;.02,.98];

        
%-------------------------------------------------------------------------%
%                       Generate observation matrix                       %
%-------------------------------------------------------------------------%

        % QR decomposition of random (uniform) noise    
        [C,~,~] = svd(randn(N,r),'econ');


%-------------------------------------------------------------------------%
%                     Generate VAR parameters                             %
%-------------------------------------------------------------------------%


        stable = false;
        snr = false;
        A = []; Q = []; theta = []; stationary = [];
        while ~stable || ~snr
            % VAR transition matrices
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

            % Observation noise covariance
            sig2Q = 5e-3;
            Q = zeros(r,r,M);
            for j = 1:M
                Q_j = randn(r);
                Q(:,:,j) = sig2Q * (Q_j' * Q_j);
            end

            theta = struct('A',A, 'C',C, 'Q',Q, 'R',R, 'mu',mu, ...
                'Sigma',Sigma, 'Pi',Pi,'Z',Z);

            stationary = get_covariance(theta,nlags,0);
            signal = sum(stationary.VAR) - sum(diag(R));
            noise = sum(diag(R));
            snr = all(signal >= 5 * noise & signal <= 10 * noise);
        end
        
        % reshape model parameters
        A = A(:);
        Q = Q(mask_Q);
        R = R(mask_R);
        Z = Z(:);
        ACF = stationary.ACF(mask_ACF);
        COV = stationary.COV(mask_COV);
        COR = stationary.COR(mask_COR);
        
 
        
%-------------------------------------------------------------------------%
%                            Simulate data                                %
%-------------------------------------------------------------------------%

       [y,S] = simulate_dyn(theta,T);
        cp = find(diff(S) ~= 0) + 1;
        
       
        
%-------------------------------------------------------------------------%
%                   Parameters for switching SSM model fit                %
%-------------------------------------------------------------------------%
        
        
        % Control parameters
        control = struct('eps',1e-6,'ItrNo',50,'beta0',1,'betarate',1,...
            'safe',false,'verbose',false);
        control2 = control; control2.ItrNo = 100;
        
        % Fixed coefficients
        fixed = [];  

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

            pars = []; pars0 = []; Zhat = []; Shat = [];
            method_name = method_list_tmp{k};
            switch method_name
            
                case 'sw-km'
        
%-------------------------------------------------------------------------%
%               Sliding windows covariance estimation                     %
%-------------------------------------------------------------------------%

            % Number of replications for k-means
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
                    
                    % Initialization (Switching SSM - OLS)
                    opts = struct('segmentation','fixed','len',50);
                    [pars0,S0] = ...
                        init_dyn(y,M,p,r,opts,control,equal,fixed,scale);
                    pars = pars0; Shat = S0;
                    
                case 'switch-ml'                    
                     try
                        [~,~,~,Shat,~,~,pars,LL] = ... 
                            switch_dyn(y,M,p,r,pars0,control,equal,fixed,scale);
                     catch
                        continue                    
                     end
                     try 
                         [~,~,pars1] = ...
                             fast_dyn(y,M,p,r,Shat,pars,control2,equal,fixed,scale);
                         [~,~,~,Shat2,~,~,pars2,LL2] = ... 
                            switch_dyn(y,M,p,r,pars1,control,equal,fixed,scale);
                         if max(LL2) > max(LL)
                             Shat = Shat2; pars = pars2;
                         end
                     catch
                         continue
                     end
                                            
                case 'or-ols'

                    % Oracle estimator (OLS)  
                    try 
                        pars = reestimate_dyn(y,M,p,r,S,control,equal,fixed,scale);
                        Shat = S;
                        pars.Z = theta.Z;
                    catch 
                        continue
                    end
                    
                case 'or-ml'
                    try 
                        % Oracle estimator (ML) 
                        [~,~,pars] = ... 
                            fast_dyn(y,M,p,r,S,pars0,control,equal,fixed,scale);
                        Shat = S;
                        pars.Z = theta.Z;
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
                HAUSDORFF_tmp(k) = max(... 
                    max(arrayfun(@(x) min(abs(x-cp)),cphat)), ...
                    max(arrayfun(@(x) min(abs(x-cphat)),cp)));
            end              

            % Classification rate
            sigma = perms(1:M); 
            Mfact = size(sigma,1); 
            classif = zeros(Mfact,1);
            for m = 1:Mfact
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
%                 Performance in covariance estimation                    %
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
                        s1(s1 < eps(1)) = 1;
                        n = numel(idx2);
                        if l == 0
                            COVhat(:,:,j) = (y(:,idx2)-m1) * (y(:,idx2)-m1)' / n;
                            CORhat(:,:,j) = (1./s1) .* COVhat(:,:,j) ./ (s1');                           
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
                stationary = get_covariance(pars,nlags,0);
                COVhat = stationary.COV;
                CORhat = stationary.COR;
                ACFhat = stationary.ACF;
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

                % Reparameterize Chat <- Chat B, Ahat <- B^-1 Ahat B, 
                % Qhat <- B^-1 Qhat (B')^-1 with B = argmin || Chat B - C ||_F  
                Ahat = pars.A; Chat = pars.C; Qhat = pars.Q; 
                Rhat = pars.R; Zhat = pars.Z; 

                try
                    B = (Chat' * Chat) \ (Chat' * C);
                catch
                    continue
                end
                for j = 1:M
                    for l = 1:p
                        Ahat(:,:,l,j) = (B\Ahat(:,:,l,j)) * B; 
                    end
                    Qhat(:,:,j) = (B\Qhat(:,:,j)) / (B');
                end
                PC = C * C';
                PChat = Chat * ((Chat'*Chat)\(Chat'));

                % Reshape estimates
                Ahat = Ahat(:);
                Qhat = Qhat(mask_Q);
                Rhat = Rhat(mask_R);
                PC = PC(mask_R);
                PChat = PChat(mask_R);

                RE_theta_tmp(k,1) = norm(Ahat-A,1) / norm(A,1);
                MSE_theta_tmp(k,1) = mean((Ahat-A).^2);
                RE_theta_tmp(k,2) = norm(PC-PChat,1) / norm(PC,1);
                MSE_theta_tmp(k,2) = mean((PC-PChat).^2);
                RE_theta_tmp(k,3) = norm(Qhat-Q,1)/norm(Q,1);
                MSE_theta_tmp(k,3) = mean((Qhat-Q).^2);
                RE_theta_tmp(k,4) = norm(Rhat-R,1)/norm(R,1);
                MSE_theta_tmp(k,4) = mean((Rhat-R).^2);
            end
            Zhat = Zhat(:);
            RE_theta_tmp(k,5) = norm(Zhat-Z,1) / norm(Z,1);
            MSE_theta_tmp(k,5) = mean((Zhat-Z).^2);
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
    

%-------------------------------------------------------------------------%
%                             Save results                                %
%-------------------------------------------------------------------------%

    clear -regexp mask
    outfile = sprintf('result_sim_dyn_N%dT%d.mat',N,T);
    save(outfile);

    
end



 

