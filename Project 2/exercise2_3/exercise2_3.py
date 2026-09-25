import numpy as np

np.set_printoptions(precision=4)

samples = {
    'ω1': np.array([
        [0.42, -0.2, 1.3, 0.39, -1.6, -0.029, -0.23, 0.27, -1.9, 0.87],
        [-0.087, -3.3, -0.32, 0.71, -5.3, 0.89, 1.9, -0.3, 0.76, -1],
        [0.58, -3.4, 1.7, 0.23, -0.15, -4.7, 2.2, -0.87, -2.1, -2.6]
    ]),
    'ω2': np.array([
        [-0.4, -0.31, 0.38, -0.15, -0.35, 0.17, 0.011, -0.27, -0.065, -0.12],
        [0.58, 0.27, 0.055, 0.53, 0.47, 0.69, 0.55, 0.61, 0.49, 0.054],
        [0.089, -0.04, -0.035, 0.011, 0.034, 0.1, -0.18, 0.12, 0.0012, -0.063]
    ])
}

omega1_x1 = np.array(samples['ω1'][0, :])
omega1_x2 = np.array(samples['ω1'][1, :])
omega1_x3 = np.array(samples['ω1'][2, :])

omega2_x1 = np.array(samples['ω2'][0, :])
omega2_x2 = np.array(samples['ω2'][1, :])
omega2_x3 = np.array(samples['ω2'][2, :])

N = np.size(omega1_x1)

# (a)
print('a) Find the mean and the covariance of omega1 using its 1D samples \n')
for i in range(3):
    mu_est = (1/N)*np.sum(samples['ω1'][i])
    cov_est = (1/N)*np.sum((samples['ω1'][i] - mu_est)**2)

    #print('The MLE for the mean of ω' + str(i+1) + ' is: %.3f' % mu_est)
    print('The MLE for the mean of ω1 given by x' + str(i+1) + ' is: %.4f' % mu_est)
    print('The MLE for the covariance of ω1 given by x' + str(i+1) + ' is: %.4f' % cov_est)
    print('-----------------------------')
print('')

# (b)
x1_x2_feature_vector = np.vstack((omega1_x1, omega1_x2)).T
x2_x3_feature_vector = np.vstack((omega1_x2, omega1_x3)).T
x3_x1_feature_vector = np.vstack((omega1_x3, omega1_x1)).T
feature_vectors = [x1_x2_feature_vector, x2_x3_feature_vector, x3_x1_feature_vector]
#print(feature_vectors)

print('b) Find the mean and the covariance matrix of omega1 using its 2D samples \n')
for i in range(3):
    sample_2D_vector = feature_vectors[i]
    #print(sample_2D_vector)
    mu_est_2D = (1/N)*np.array([np.sum([samples['ω1'][i%3]]), np.sum([samples['ω1'][(i+1)%3]])])

    #sample_minus_mu = np.array([np.diff([feature_vectors[i,:]], [mu_est_2D], axis = 0)])
    #Sigma_est_2D = (1/N)*np.sum(np.dot(x1_x2_feature_vector))
    #sample_2D_vector = feature_vectors[i]
    #print(sample_2D_vector)
    #print(mu_est_2D)
    #sample_minus_mu = sample_2D_vector[:] - mu_est_2D
    #print(sample_minus_mu)
    sample_minus_mu = sample_2D_vector - mu_est_2D
    cov_est_2D = (1/N) * np.dot(sample_minus_mu.T, sample_minus_mu)
    print('The MLE for the mean of ω1 given by x' + str((i+1)%4) + ',x' + str((i+2)%4) + ' is: \n' + str(np.transpose(mu_est_2D)) )
    print('The MLE for the covariance matrix of ω1 given by x' + str((i+1)%4) + ',x' + str((i+2)%4) + ' is: \n' + str(cov_est_2D))
    print('-----------------------------')
print('')

# (c)
print('c) Find the mean and the covariance matrix of omega1 using its 3D samples \n')

sample_3D_vector = np.vstack(([samples['ω1'][0]], [samples['ω1'][(1)]], [samples['ω1'][(2)]])).T
mu_est_3D = (1/N)*np.array([np.sum([samples['ω1'][0]]), np.sum([samples['ω1'][(1)]]), np.sum([samples['ω1'][(2)]])])

sample_minus_mu = sample_3D_vector - mu_est_3D
cov_est_3D = (1/N) * np.dot(sample_minus_mu.T, sample_minus_mu)
print('The MLE for the mean of ω1 given by its 3D samples is: \n' + str(np.transpose(mu_est_3D)) )
print('The MLE for the covariance matrix of ω1 given by its 3D samples is: \n' + str(cov_est_3D))
print('-----------------------------')

# (d)
print('d) Find the mean and the covariance matrix of omega2 using its 3D samples \n')
m = np.zeros(3)
s = np.zeros(3)
for i in range(3):
    mu_est = (1/N)*np.sum(samples['ω2'][i])
    cov_est = (1/N)*np.sum((samples['ω2'][i] - mu_est)**2)

    m[i] = mu_est
    s[i] = cov_est
    if i == 2:
        cov_matrix = np.diag(s)

        print('The MLE for the mean of ω1 given by x' + str(i+1) + ' is: ' + str(m))
        print('The MLE for the covariance of ω1 given by x' + str(i+1) + ' is: \n' + str(cov_matrix))
print('-----------------------------')
print('')



