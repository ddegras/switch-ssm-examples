function [S,Z] = swkm(y,M,len,nrep)

narginchk(3,4)

% Check that window length is valid
assert(len >= 1)
assert(len <= size(y,2))

if nargin == 3
   nrep = 100;
end  

% Round window width up or down as needed
lb = floor(len);
if mod(lb,2) == 1 
    len = lb;
else
    len = lb + 1;
end
    
[N,T] = size(y); % data dimensions
halflen = floor(len/2); % half length
V = zeros(N,N,T); % sliding covariance matrices

% Boundary adjustments
V(:,:,1:halflen+1) = repmat(cov(y(:,1:len)',1),1,1,halflen+1);
V(:,:,T-halflen+1:T) = repmat(cov(y(:,T-len+1:T)',1),1,1,halflen);

% S = [];
% Z = [];

% Sliding covariance
if len < T
    sumy = sum(y(:,1:len),2);
    sumyy = y(:,1:len) * y(:,1:len)';
    for t = halflen+2:T-halflen 
        sumy = sumy - y(:,t-halflen-1) + y(:,t+halflen);
        sumyy = sumyy - (y(:,t-halflen-1) * y(:,t-halflen-1)') + ...
            (y(:,t+halflen) * y(:,t+halflen)');
        V(:,:,t) = sumyy/len - (sumy * sumy')/(len^2);
    end
end

% V2 = V;
% for t = halflen+2:T-halflen
%     V2(:,:,t) = cov(y(:,t-halflen:t+halflen)',1);
% end
 

% K-means clustering
mask = logical(tril(ones(N)));
mask = mask(:);
V = reshape(V,[],T);
V = V(mask,:)';

[S,~,~,~] = kmeans(V,M,'Distance','cityblock','Replicates',nrep);
S = S';

% Transition probability matrix
Z = NaN(M);
for i = 1:M
    nrm = sum(S(1:T-1) == i);
    if nrm == 0
        Z(i,:) = 1/M;
        continue
    end
    for j = 1:M
        Z(i,j) = sum(S(1:T-1) == i & S(2:T) == j) / nrm;
    end
end
                    