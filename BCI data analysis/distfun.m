function dAB = distfun(A,wA,B,wB) 

% DISTFUN measures the weighted distance between two sets of vectors A and B. 
% Given A = {a(1),...,a(m)} and B = {b(1),...,b(n)} and weights w(i,A), i=1:m, 
% and w(j,B), j=1:n, the distance is defined as 
% d = 0.5 * sum(i=1:m) min(j=1:n) w(i,A) || a(i) - b(j) ||^2 ... 
% + 0.5 * sum(j=1:n) min(i=1:m) w(j,B) || a(i) - b(j) ||^2
% If a single set A is provided, the weighted variance 
% (1-sum(i)(w(i,A)^2)) / sum(i,j) w(i,A) w(j,A) || a(i) - a(j) ||^2
% is returned

assert(ismatrix(A))
m = size(A,2); 
if ~exist('wA','var')
    wA = ones(1,m);
end
wA = wA(:);
assert(numel(wA) == size(A,2))
assert(all(wA >= 0) && sum(wA) > 0);
wA = wA / sum(wA);

if exist('B','var') && ~isempty(B)
    assert(ismatrix(B))
    assert(size(A,1) == size(B,1))
    n = size(B,2);
    if ~exist('wB','var')
        wB = ones(1,n);
    end
    wB = wB(:);
    assert(numel(wB) == size(B,2))
    assert(all(wB >= 0) && sum(wB) > 0);
    wB = wB / sum(wB);
    
    % Distances between individual vectors
    D = zeros(m,n); 
    for i = 1:m
        for j = 1:n
            D(i,j) = sum((A(:,i)-B(:,j)).^2);
        end
    end

    % Distance between sets
    dist1 = min(wA .* D,1);
    dist2 = min(D .* (wB'),2);
    dAB = 0.5 * sum(dist1) + 0.5 * sum(dist2); 

else
    nA = (wA') .* sum(A.^2);
    dAB = 2 * (sum(nA) - norm(A * wA)^2) / (1 - norm(wA)^2); 
end













