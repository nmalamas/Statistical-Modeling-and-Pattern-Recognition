%% Pattern Recognition 2024
%  Exercise 1.4 | Bayes

clear all; close all; clc

% 2.
% Define the parameters of the two Gaussian functions
mu1 = [2 3];
mu2 = [4 4];
Sigma1 = [2 0.5; 0.5 1];
Sigma2 = [1.5 -0.3; -0.3 0.8];

% Define the x and y values for the plot
x= linspace(-2, 7.5, 100);
y= linspace(0, 6.5, 100);
[X,Y] = meshgrid(x,y);

% Create the Gaussian class omega 1
omega1 = mvnpdf([X(:) Y(:)], mu1, Sigma1);
omega1 = reshape(omega1, size(X));

% Create the Gaussian class omega 2
omega2 = mvnpdf([X(:) Y(:)], mu2, Sigma2);
omega2 = reshape(omega2, size(X));

% Project the Gaussian functions onto the x-y plane
% Contour
figure
contour(X,Y,omega1,20,'g');
hold on;
contour(X,Y,omega2,20,'b');
axis tight;
xlabel('x');
ylabel('y');
legend('\omega_1 class','\omega_2 class')
title('The PDFs of \omega_1 and \omega_2 classes');

% Surf
figure
surf(X,Y,omega1,'FaceColor','g');
hold on;
surf(X,Y,omega2,'FaceColor','b');
%colorbar;
axis tight;
xlabel('x');
ylabel('y');
legend('\omega_1 class','\omega_2 class')
title('The PDFs of \omega_1 and \omega_2 classes');

% 3.
figure

% For p1=0.1
p1 = 0.1;
p2 = 1 - p1;
const1 = 6.27 - 2*log(1.57*(p2/p1));
dec_bd1 = 0.14.*X.^2 + 0.20.*Y.^2 - 6.66.*X - 7.25.*Y + 1.11.*X.*Y + 2*log(p2/p1) - log(det(Sigma2)/det(Sigma1)) - 32.65;
surf(X,Y,dec_bd1,'FaceColor','r','EdgeColor','k');
hold on;

% For p1=0.25
p1 = 0.25;
p2 = 1 - p1;
const2 = 6.27 - 2*log(1.57*(p2/p1));
dec_bd2 = 0.14.*X.^2 + 0.20.*Y.^2 - 6.66.*X - 7.25.*Y + 1.11.*X.*Y + 2*log(p2/p1) - log(det(Sigma2)/det(Sigma1)) - 32.65;
surf(X,Y,dec_bd2,'FaceColor','b','EdgeColor','k');
hold on;

% For p1=0.5
p1 = 0.5;
p2 = 1 - p1;
const2 = 6.27 - 2*log(1.57*(p2/p1));
dec_bd3 = 0.14.*X.^2 + 0.20.*Y.^2 - 6.66.*X - 7.25.*Y + 1.11.*X.*Y + 2*log(p2/p1) - log(det(Sigma2)/det(Sigma1)) - 32.65;
surf(X,Y,dec_bd3,'FaceColor','c','EdgeColor','k');
hold on;

% For p1=0.75
p1 = 0.75;
p2 = 1 - p1;
const4 = 32.58 - log(0.16*(p2/p1));
dec_bd4 = 0.14.*X.^2 + 0.20.*Y.^2 - 6.66.*X - 7.25.*Y + 1.11.*X.*Y + 2*log(p2/p1) - log(det(Sigma2)/det(Sigma1)) - 32.65;
surf(X,Y,dec_bd4,'FaceColor','m','EdgeColor','k');
hold on;

% For p1=0.9
p1 = 0.9;
p2 = 1 - p1;
const5 = 32.58 - log(0.16*(p2/p1));
dec_bd5 = 0.14.*X.^2 + 0.20.*Y.^2 - 6.66.*X - 7.25.*Y + 1.11.*X.*Y + 2*log(p2/p1) - log(det(Sigma2)/det(Sigma1)) - 32.65;
surf(X,Y,dec_bd5,'FaceColor','y','EdgeColor','k');
hold on;

xlabel('x');
ylabel('y');
zlabel('z');
title('Decision boundary for different values of p(\omega_1)');
legend('Dec. Bd for p(\omega_1) = 0.1','Dec. Bd for p(\omega_1) = 0.25','Dec. Bd for p(\omega_1) = 0.5','Dec. Bd for p(\omega_1) = 0.75','Dec. Bd for p(\omega_1) = 0.9');

% Plotting the surfce once more, using 'interp', 
% so that the shape of the boundary is clear
figure

surf(X,Y,dec_bd1,'EdgeColor','interp');
hold on;
surf(X,Y,dec_bd2,'EdgeColor','interp');
hold on;
surf(X,Y,dec_bd3,'EdgeColor','interp');
hold on;
surf(X,Y,dec_bd4,'EdgeColor','interp');
hold on;
surf(X,Y,dec_bd5,'EdgeColor','interp');
hold on;

xlabel('x');
ylabel('y');
zlabel('z');
title('Decision boundary for different values of p(\omega_1)');
legend('Dec. Bd for p(\omega_1) = 0.1','Dec. Bd for p(\omega_1) = 0.25','Dec. Bd for p(\omega_1) = 0.5','Dec. Bd for p(\omega_1) = 0.75','Dec. Bd for p(\omega_1) = 0.9');

% 4.
% Repeat for the given correlation matrices (Sigma1 = Sigma2)

% b.
% Define the parameters of the two Gaussian functions
mu1 = [2 3];
mu2 = [4 4];
Sigma1 = [1.2 0.4; 0.4 1.2];
Sigma2 = Sigma1;

% Define the x and y values for the plot
x= linspace(-1, 7, 100);
y= linspace(0, 7, 100);
[X,Y] = meshgrid(x,y);

% Create the Gaussian class omega 1
omega1 = mvnpdf([X(:) Y(:)], mu1, Sigma1);
omega1 = reshape(omega1, size(X));

% Create the Gaussian class omega 2
omega2 = mvnpdf([X(:) Y(:)], mu2, Sigma2);
omega2 = reshape(omega2, size(X));

% Project the Gaussian functions onto the x-y plane
% Contour
figure
contour(X,Y,omega1,20,'g');
hold on;
contour(X,Y,omega2,20,'b');
axis tight;
xlabel('x');
ylabel('y');
legend('\omega_1 class','\omega_2 class')
title('The PDFs of \omega_1 and \omega_2 classes');

% Surf
figure
surf(X,Y,omega1,'FaceColor','g');
hold on;
surf(X,Y,omega2,'FaceColor','b');
axis tight;
xlabel('x');
ylabel('y');
legend('\omega_1 class','\omega_2 class')
title('The PDFs of \omega_1 and \omega_2 classes');

% c.
mu1 = [2 3];
mu2 = [4 4];
Sigma1 = [1.2 0.4; 0.4 1.2];
Sigma2 = Sigma1;

figure

% For p1=0.1
p1 = 0.1;
p2 = 1 - p1;
const1 = 11.56 - 2*log((p2/p1));
x1 = x;
x2 = (-1/0.62)*(3.12.*x1-const1);
plot(x1,x2,'r');
hold on;

% For p1=0.25
p1 = 0.25;
p2 = 1 - p1;
const2 = 11.56 - 2*log((p2/p1));
x1 = x;
x2 = (-1/0.62)*(3.12.*x1-const2);
plot(x1,x2,'y');
hold on;

% For p1=0.5
p1 = 0.5;
p2 = 1 - p1;
const3 = 11.56 - 2*log((p2/p1));
x1 = x;
x2 = (-1/0.62)*(3.12.*x1-const3);
plot(x1,x2,'g');
hold on;

% For p1=0.75
p1 = 0.75;
p2 = 1 - p1;
const4 = 11.56 - 2*log((p2/p1));
x1 = x;
x2 = (-1/0.62)*(3.12.*x1-const4);
plot(x1,x2,'b');
hold on;

% For p1=0.9
p1 = 0.9;
p2 = 1 - p1;
const5 = 11.56 - 2*log((p2/p1));
x1 = x;
x2 = (-1/0.62)*(3.12.*x1-const5);
plot(x1,x2,'c');
hold on;

xlabel('x');
ylabel('y');
title('Decision boundary for different values of p(\omega_1)');
legend('Dec. Bd for p(\omega_1) = 0.1','Dec. Bd for p(\omega_1) = 0.25', ...
    'Dec. Bd for p(\omega_1) = 0.5','Dec. Bd for p(\omega_1) = 0.75','Dec. Bd for p(\omega_1) = 0.9');