function out = fit_bci(y,model,M,p,r,opts,control,control2,ncycles) 

% Data dimensions
[N,T] = size(y);

% Handles for EM functions
switch model 
    case 'dyn'
        init_fun = @init_dyn;
        em_fun = @switch_dyn;
        fast_fun = @fast_dyn;
    case 'obs'
        init_fun = @init_obs;
        em_fun = @switch_obs;
        fast_fun = @fast_obs;
end        

% Initialization
if ~exist('opts','var')
    opts = [];
end
if ~exist('control','var')
    control = [];
end
verbose = true;
if isstruct(control) && isfield(control,'verbose')
    verbose = control.verbose;
end
if ~exist('control2','var')
    control2 = [];
end
verbose2 = true;
if isstruct(control2) && isfield(control2,'verbose')
    verbose2 = control2.verbose;
end
verbose2 = verbose && verbose2; 
if ~exist('ncycles','var')
    ncycles = 5;
end
pars = init_fun(y,M,p,r,opts);
LLbest = -Inf;
LL = [];

% Main loop  
for i = 1:ncycles
    if verbose 
        fprintf('\n\nCycle %d\n\n',i);
    end
    LLbestold = LLbest;
    if i == 2 && isstruct(control) 
        control.beta0 = 1;
    end
    
    % Regular EM
    [~,~,~,Shat,~,~,pars,LLtmp] = em_fun(y,M,p,r,pars,control); 
    
    % Keep track of (log-)likelihood history
    LL = [LL ; LLtmp(:)];  %#ok<AGROW>
    
    % Check progress criterion. Stop if insufficient progress, store best
    % performance to date
    if max(LLtmp) <= (1+control.eps) * LLbestold
        break
    else
        parsbest = pars; LLbest = max(LLtmp); Sbest = Shat;
    end   
    
    % Acceleration: fit EM with regimes fixed to most likely values 
     if verbose2
        fprintf('\n\n');  
    end
    [~,~,pars] = fast_fun(y,M,p,r,Shat,pars,control2);
   
end

% AIC and BIC scores
switch model
    case 'dyn'
        PEN = N*r + M*p*r^2 + M*r*(r+1)/2 + N*(N+1)/2 + r + r*(r+1)/2 + (p-1)*(p+1); 
    case 'obs'
        PEN = M*N*r + M*p*r^2 + M*r*(r+1)/2 + N*(N+1)/2 + r + r*(r+1)/2 + (p-1)*(p+1); 
end       
AIC = -2 * LLbest + 2 * PEN; 
BIC = -2 * LLbest + log(T) * PEN;


% Pur results together in structure
out = struct('model',model,'M',M,'p',p,'r',r,'pars',parsbest,...
    'regime',Sbest,'LL',LL,'LLbest',LLbest,'AIC',AIC,'BIC',BIC); 
