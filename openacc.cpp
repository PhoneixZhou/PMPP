while(err > tol && iter < iter_max){
    err = 0.0;
    #pragma acc kernels
    {
        for(int j = 1;j<n-1;j++){
            for(int i = 1;i<m-1;i++){
                Anew[j][i] = 0.25 * (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
                err = max(err,abs(Anew[j][i] - A[j][i]));
            }
        }
        
        for(int j = 1;j<n-1;j++){
            for(int i = 1;i < m - 1;i++){
                A[j][i] = Anew[j][i];
            }
        }
    }
    iter++;
}