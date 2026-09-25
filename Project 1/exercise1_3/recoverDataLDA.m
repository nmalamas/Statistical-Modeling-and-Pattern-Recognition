function [X_rec] = recoverDataLDA(Z, v)

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), length(v));

% ====================== YOUR CODE HERE ======================

% for i=1:length(Z)
%     X_rec(i,:) = v' * Z(i,:)';
% end
% 
% X_rec(i,:) = X_rec(i,:)/norm(v);

X_rec = Z * v';
% =============================================================

end
