data {
  int <lower=0> K; # number of images
  int <lower=0> M; # height/width of each image
  int <lower=0> N; # number of pixels in superimage
  vector[M] y[K]; # the input images
  matrix[N, N] Zx; # covariance matrix (in terms of A and r)
  vector[2] v_bar;
  vector[2] vi[N];
  vector[2] vj[M];
  real beta; # noise 
}

parameters {
  real gamma;
  real theta[K];
  vector[2] s[K];
}

transformed parameters {
  matrix[2, 2] R[K];
  vector[2] u[K, M];
  matrix[M, N] W_tilde[K];
  matrix[K, M] W_sum;
  matrix[M, N] W[K];
  matrix[N, N] sigma_interm;
  matrix[N, N] sigma;
  vector[N] mu_interm;
  vector[N] mu;

  # define R, the rotation matrix
  for (i in 1:K) {
    R[i][0][0]= cos(theta[i]);
    R[i][0][1]= sin(theta[i]);
    R[i][1][0]= -sin(theta[i]);
    R[i][1][1]= cos(theta[i]);
  }
 
  # define u
  for (k in 1:K) {
    for (j in 1:M) { 
      u[k][j] = R[k]*(vj[j] - v_bar) + v_bar + s[k];
    }
  }

  # define W_tilde
  for (k in 1:K) {
    for (j in 1:M) {
      for (i in 1:N) {
        W_tilde[k][j][i] = exp(-squared_distance(vi[i], u[k][j])/pow(gamma, 2));
      }
    }
  }

  # define W_sum (used to calculate W)
  W_sum = rep_matrix(0, K, M);
  for (k in 1:K) {
    for (j in 1:M) {
      for (i in 1:N) {
        W_sum[k][j] = W_sum[k][j] + W_tilde[k][j][i];
      }
    }
  }

  # define W 
  for (k in 1:K) {
    for (j in 1:M) {
      for (i in 1:N) {
        W[k][j][i] = W_tilde[k][j][i]/W_sum[j][j];
      }
    }
  }

  # define sigma_interm (used to find sigma)
  sigma_interm = rep_matrix(0, N, N);
  for (k in 1:K) {
    sigma_interm = sigma_interm + transpose(W[k])*W[k];
  }

  # define sigma
  sigma = inverse(inverse(Zx) + beta*sigma_interm); 

  # define mu_imterm (used to find mu)
  mu_interm = rep_vector(0, N);
  for (k in 1:K) {
    mu_interm = mu_interm + transpose(W[k])*y[k];
  }

  # define mu
  mu = beta * sigma * mu_interm;
}

model {
  real likelihood;
  likelihood = 0;
  for (k in 1:K) {
    likelihood = likelihood + beta*squared_distance(y[k], W[k]*mu);
  }
  likelihood = likelihood + transpose(mu)*inverse(Zx)*mu;
  likelihood = likelihood + log(determinant(Zx)) - log(determinant(sigma)) - K*M*log(beta);
  likelihood = -0.5*likelihood;
}

