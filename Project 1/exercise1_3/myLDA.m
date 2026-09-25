function [A] = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
    [NumSamples NumFeatures] = size(Samples);
    NumLabels = length(Labels);

	A=zeros(NumFeatures,NewDim);
	
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes)  %The number of classes

    Sw = zeros(length(NumClasses));
    Sb = zeros(length(NumClasses));
    mu = zeros(length(NumClasses), NumFeatures);
    P = zeros(length(NumClasses));
    m0=0;
    %For each class i
	%Find the necessary statistics
    for i=1:NumClasses
        %Calculate the Class Prior Probability
	    P(i)=length(Samples(Labels==i-1))/NumSamples;
        
        %Calculate the Class Mean 
	    mu(i,:)=mean(Samples(Labels==i-1,:));
        
        %Calculate the Within Class Scatter Matrix
	    Sw=Sw+P(i)*cov(Samples(Labels==i-1,:));
        
        %Calculate the Global Mean
	    m0=mean(mu);

        %Calculate the Between Class Scatter Matrix
	    Sb=Sb+ P(i)*(mu(i,:)-m0)'*(mu(i,:)-m0);
    end 
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = Sw^(-1)*Sb;
    
    %Perform Eigendecomposition
    %Correlation autocorrelation matrix
    SigmaCorr = (1/NumSamples)*EigMat'*EigMat;

    % Find the eigenvalues and eigenvectors of the autocorrelation matrix:
    [V,D] = eig(EigMat);

    % Vector of eigvalues
    eigenval = diag(D);

    order = 'descend';

    [eigenval,ind]=sort(eigenval,NewDim,order);
    eigenvec=V(:,1:NewDim);
   
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A(1:end,:)=eigenvec(:,1:NewDim);  % Return the LDA projection vectors
end
