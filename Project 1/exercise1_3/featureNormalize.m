function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


% ADD YOUR CODE
% Find the mean value of X, mu as:
mu = mean(X);

% Find the standard deviation, sigma as:
sigma = std(X);

% The standardized variable is:
X_norm = (X - mu)./sigma;


% ============================================================

end
