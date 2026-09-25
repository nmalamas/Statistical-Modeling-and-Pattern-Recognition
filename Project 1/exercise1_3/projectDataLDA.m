function [Z] = projectDataLDA(X, v)

% You need to return the following variables correctly.
Z = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================

% for i=1:length(X)
%     Z(i,:) = v' * X(i,:)';
% end
% 
% Z(i,:) = Z(i,:)/norm(v);

Z = X*v;
% =============================================================

end
