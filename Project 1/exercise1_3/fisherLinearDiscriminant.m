function [v] = fisherLinearDiscriminant(X1, X2)

    m1 = size(X1, 1);
    m2 = size(X2, 1);

    % mean value of X1
    mu1 = mean(X1);
    % mean value of X2
    mu2 = mean(X2);

    % scatter matrix of X1
    S1 = cov(X1);
    % scatter matrix of X2
    S2 = cov(X2);

    X = [X1;X2];
    for i=1:2
        P(i)=length(X((i-1)*m1+1:i*m1,:))/(m1+m2);
    end

    % Within class scatter matrix
    Sw = P(1)*S1 + P(2)*S2;

    % optimal direction for maximum class separation 
    v = Sw^(-1)*(mu1 - mu2)';
    % return a vector of unit norm
    v = v/norm(v);
