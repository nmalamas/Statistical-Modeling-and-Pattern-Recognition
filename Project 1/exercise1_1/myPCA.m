function [ eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [ eigenval, eigenvec, order] = mypca(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order 
%

% Useful values
[m, n] = size(X);

% Make sure each feature from the data is zero mean
X_centered = X - mean(X);

% ====================== YOUR CODE HERE ======================
%

% The autocorrelation matrix Sigma is given as:
SigmaCorr = (1/m)*(X_centered)'*X_centered;

% Find the eigenvalues and eigenvectors of the autocorrelation matrix:
[V,D] = eig(SigmaCorr);

eigenval=diag(D); %Vector of eigenvalues
order = 'descend'; % Order of importance, in PCA

[eigenval,ind]=sort(eigenval,1,order); %Sort the eigenvalues in descending order
eigenvec=V(:,ind); % corresponding eigenvectors

% =========================================================================

end
